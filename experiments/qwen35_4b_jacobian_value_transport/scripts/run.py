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

from io_utils import read_jsonl, sha256_file, write_json, write_jsonl  # noqa: E402
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


def run_fit_lens(config: dict[str, Any]) -> dict[str, Any]:
    design = design_boundary_receipt(config)
    if not (DATA_DIR / "manifest.json").exists():
        build_splits(DATA_DIR, config)
    import time
    import torch
    from model_ops import QwenTransportModel

    started = time.perf_counter()
    model = QwenTransportModel(config)
    concepts = tuple(CONCEPT_CANDIDATES)
    corpus = [row["text"] for row in read_jsonl(DATA_DIR / "lens_fit.jsonl")]
    lens_config = config["lens"]
    lens, prompt_receipts = model.fit_targeted_lens(
        corpus,
        concepts,
        source_layers=tuple(int(value) for value in lens_config["source_layers"]),
        target_layer=int(lens_config["target_layer"]),
        concept_batch=int(lens_config["dimension_batch"]),
        max_sequence_tokens=int(lens_config["max_sequence_tokens"]),
        skip_first_positions=int(lens_config["skip_first_positions"]),
    )
    artifact = RUNS_DIR / "targeted_lens.pt"
    torch.save(lens.state_dict(), artifact)
    norms = {str(layer): lens.directions[layer].norm(dim=1).tolist() for layer in lens.source_layers}
    result = {
        "schema_version": 1,
        "passed": bool(
            design["passed"]
            and lens.n_prompts == len(corpus)
            and all(torch.isfinite(value).all() and bool((value.norm(dim=1) > 0).all()) for value in lens.directions.values())
        ),
        "model_revision": config["model"]["revision"],
        "concepts": list(lens.concepts),
        "token_ids": list(lens.token_ids),
        "source_layers": list(lens.source_layers),
        "target_layer": lens.target_layer,
        "n_prompts": lens.n_prompts,
        "pair_weighting": lens.pair_weighting,
        "direction_norms": norms,
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
    if not result["passed"]:
        raise RuntimeError(f"targeted lens fit failed: {result}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _random_direction_pairs(batch: int, width: int, *, seed: int) -> tuple["torch.Tensor", "torch.Tensor"]:
    import torch

    generator = torch.Generator().manual_seed(seed)
    matrices = torch.randn(batch, width, 2, generator=generator)
    q, _r = torch.linalg.qr(matrices, mode="reduced")
    return q[:, :, 0], q[:, :, 1]


def _score_control_batch(
    model,
    lens,
    rows: list[dict[str, Any]],
    *,
    prompt_kind: str,
    condition: str,
    layers: tuple[int, ...],
    alpha: float,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    import torch

    concept_index = {concept: index for index, concept in enumerate(lens.concepts)}
    if prompt_kind == "direct":
        prefixes = [model.render(row["direct_prompt"], enable_thinking=False) + "Concept:" for row in rows]
        source_ids = [model.concept_token_id(row["source"]) for row in rows]
        target_ids = [model.concept_token_id(row["target"]) for row in rows]
        parse_ids = set(lens.token_ids)
    elif prompt_kind == "consequence":
        prefixes = [model.render(row["consequence_prompt"], enable_thinking=False) + "Value:" for row in rows]
        source_ids = [model.concept_token_id(str(row["source_value"])) for row in rows]
        target_ids = [model.concept_token_id(str(row["target_value"])) for row in rows]
        parse_ids = {model.concept_token_id(str(value)) for value in range(10)}
    else:
        raise ValueError(prompt_kind)

    layer_directions = None
    if condition != "baseline":
        layer_directions = {}
        for layer in layers:
            if condition == "j":
                source = torch.stack([lens.directions[layer][concept_index[row["source"]]] for row in rows])
                target = torch.stack([lens.directions[layer][concept_index[row["target"]]] for row in rows])
            elif condition == "j_wrong":
                source = torch.stack([lens.directions[layer][concept_index[row["source"]]] for row in rows])
                target = torch.stack([
                    lens.directions[layer][(concept_index[row["target"]] + 1) % len(lens.concepts)]
                    for row in rows
                ])
            elif condition == "logit":
                source = model.lm_head.weight[[model.concept_token_id(row["source"]) for row in rows]].float().cpu()
                target = model.lm_head.weight[[model.concept_token_id(row["target"]) for row in rows]].float().cpu()
            elif condition == "random":
                source, target = _random_direction_pairs(
                    len(rows), model.d_model,
                    seed=int(config["seeds"]["controls"]) + layer * 1009 + sum(ord(c) for c in prompt_kind),
                )
            else:
                raise ValueError(condition)
            layer_directions[layer] = (source, target)
    scored = model.score_next_token_batch(prefixes, layer_directions=layer_directions, alpha=alpha)
    output = []
    for index, row in enumerate(rows):
        logits = scored["logits"][index]
        top_id = int(scored["top_ids"][index])
        output.append({
            "item_id": row["item_id"],
            "split_half": "selection" if int(row["item_id"].split("-")[-1]) % 2 == 0 else "confirmation",
            "prompt_kind": prompt_kind,
            "condition": condition,
            "layers": list(layers),
            "alpha": alpha,
            "source": row["source"],
            "target": row["target"],
            "source_id": source_ids[index],
            "target_id": target_ids[index],
            "top_id": top_id,
            "top_text": model.tokenizer.decode([top_id]),
            "source_correct": top_id == source_ids[index],
            "target_selected": top_id == target_ids[index],
            "parsed": top_id in parse_ids,
            "target_minus_source_logit": float(logits[target_ids[index]] - logits[source_ids[index]]),
            "delta_norm": float(scored["delta_norms"][index]),
            "sequence_tokens": scored["sequence_tokens"],
            "forward_tokens": scored["sequence_tokens"],
        })
    return output


def _summarize_control(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize empty control rows")
    return {
        "n": len(rows),
        "source_accuracy": sum(bool(row["source_correct"]) for row in rows) / len(rows),
        "target_rate": sum(bool(row["target_selected"]) for row in rows) / len(rows),
        "parse_rate": sum(bool(row["parsed"]) for row in rows) / len(rows),
        "mean_margin": sum(float(row["target_minus_source_logit"]) for row in rows) / len(rows),
        "mean_delta_norm": sum(float(row["delta_norm"]) for row in rows) / len(rows),
    }


def run_positive_control(config: dict[str, Any]) -> dict[str, Any]:
    design = design_boundary_receipt(config)
    import time
    from model_ops import QwenTransportModel, TargetedLens

    lens_path = RUNS_DIR / "targeted_lens.pt"
    if not lens_path.exists():
        raise RuntimeError("targeted lens is missing; run --stage fit-lens first")
    started = time.perf_counter()
    model = QwenTransportModel(config)
    lens = TargetedLens.load(str(lens_path))
    all_items = read_jsonl(DATA_DIR / "positive_control.jsonl")
    selection = [row for index, row in enumerate(all_items) if index % 2 == 0]
    confirmation = [row for index, row in enumerate(all_items) if index % 2 == 1]
    layers = tuple(int(value) for value in config["lens"]["source_layers"])
    alphas = tuple(float(value) for value in config["lens"]["coordinate_alphas"])
    result_rows: list[dict[str, Any]] = []

    # Baseline is coefficient-independent.
    for kind in ("direct", "consequence"):
        result_rows += _score_control_batch(
            model, lens, selection, prompt_kind=kind, condition="baseline",
            layers=(), alpha=0.0, config=config,
        )
    for alpha in alphas:
        for layer in layers:
            for kind in ("direct", "consequence"):
                for condition in ("j", "random", "logit", "j_wrong"):
                    result_rows += _score_control_batch(
                        model, lens, selection, prompt_kind=kind, condition=condition,
                        layers=(layer,), alpha=alpha, config=config,
                    )

    def subset(source_rows, **matches):
        return [row for row in source_rows if all(row[key] == value for key, value in matches.items())]

    baseline_selection = {
        kind: _summarize_control(subset(result_rows, split_half="selection", prompt_kind=kind, condition="baseline"))
        for kind in ("direct", "consequence")
    }
    candidates = []
    for alpha in alphas:
        for left, right in zip(layers[:-1], layers[1:], strict=True):
            per_layer = []
            for layer in (left, right):
                cell = {}
                for kind in ("direct", "consequence"):
                    j = _summarize_control(subset(
                        result_rows, split_half="selection", prompt_kind=kind,
                        condition="j", layers=[layer], alpha=alpha,
                    ))
                    random = _summarize_control(subset(
                        result_rows, split_half="selection", prompt_kind=kind,
                        condition="random", layers=[layer], alpha=alpha,
                    ))
                    cell[kind] = {"j": j, "random": random}
                per_layer.append(cell)
            score = sum(
                cell[kind]["j"]["target_rate"] - cell[kind]["random"]["target_rate"]
                for cell in per_layer for kind in ("direct", "consequence")
            )
            candidates.append({"alpha": alpha, "layers": [left, right], "score": score, "cells": per_layer})
    selected = max(candidates, key=lambda row: (row["score"], -row["alpha"], -row["layers"][0]))

    # Confirmation runs the selected coefficient at every individual layer plus
    # the selected adjacent band, with all controls untouched by selection labels.
    confirm_layer_sets = [(layer,) for layer in layers] + [tuple(selected["layers"])]
    for layer_set in confirm_layer_sets:
        for kind in ("direct", "consequence"):
            for condition in ("j", "random", "logit", "j_wrong"):
                result_rows += _score_control_batch(
                    model, lens, confirmation, prompt_kind=kind, condition=condition,
                    layers=layer_set, alpha=float(selected["alpha"]), config=config,
                )
    for kind in ("direct", "consequence"):
        result_rows += _score_control_batch(
            model, lens, confirmation, prompt_kind=kind, condition="baseline",
            layers=(), alpha=0.0, config=config,
        )

    gates = config["gates"]["positive_control"]
    confirmation_summary = {}
    passing_individual = []
    baseline_confirmation = {
        kind: _summarize_control(subset(result_rows, split_half="confirmation", prompt_kind=kind, condition="baseline"))
        for kind in ("direct", "consequence")
    }
    for layer in layers:
        cell = {}
        passes = True
        for kind, shift_min in (
            ("direct", float(gates["verbal_target_shift_min"])),
            ("consequence", float(gates["consequence_target_shift_min"])),
        ):
            j = _summarize_control(subset(
                result_rows, split_half="confirmation", prompt_kind=kind,
                condition="j", layers=[layer], alpha=float(selected["alpha"]),
            ))
            random = _summarize_control(subset(
                result_rows, split_half="confirmation", prompt_kind=kind,
                condition="random", layers=[layer], alpha=float(selected["alpha"]),
            ))
            shift = j["target_rate"] - baseline_confirmation[kind]["target_rate"]
            advantage = j["target_rate"] - random["target_rate"]
            parse_drop = baseline_confirmation[kind]["parse_rate"] - j["parse_rate"]
            cell[kind] = {"j": j, "random": random, "target_shift": shift, "j_minus_random": advantage, "parse_drop": parse_drop}
            passes = bool(
                passes
                and shift >= shift_min
                and advantage >= float(gates["j_minus_random_min"])
                and parse_drop <= float(gates["max_parse_rate_drop"])
            )
        confirmation_summary[str(layer)] = cell
        if passes:
            passing_individual.append(layer)
    adjacent_pass = any(
        left in passing_individual and right in passing_individual
        for left, right in zip(layers[:-1], layers[1:], strict=True)
    )
    clean_pass = all(
        baseline_confirmation[kind]["source_accuracy"] >= float(gates["clean_accuracy_min"])
        for kind in ("direct", "consequence")
    )
    gate_pass = bool(design["passed"] and clean_pass and adjacent_pass)
    result = {
        "schema_version": 1,
        "passed": gate_pass,
        "decision": "POSITIVE_CONTROL_PASS" if gate_pass else "NO_J_WRITING",
        "selection": selected,
        "baseline_selection": baseline_selection,
        "baseline_confirmation": baseline_confirmation,
        "confirmation_by_layer": confirmation_summary,
        "passing_individual_layers": passing_individual,
        "adjacent_pass": adjacent_pass,
        "clean_pass": clean_pass,
        "counts": {"selection_items": len(selection), "confirmation_items": len(confirmation), "rows": len(result_rows)},
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_jsonl(RUNS_DIR / "positive_control_rows.jsonl", result_rows)
    write_json(RUNS_DIR / "positive_control.json", result)
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
    if args.stage == "fit-lens":
        run_fit_lens(config)
        return 0
    if args.stage == "positive-control":
        run_positive_control(config)
        return 0
    design_boundary_receipt(config)
    unavailable_stage(args.stage)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

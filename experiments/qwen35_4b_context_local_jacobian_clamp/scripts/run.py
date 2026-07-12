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

from io_utils import read_json, read_jsonl, sha256_file, write_json, write_jsonl  # noqa: E402
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
    single_top_ids = [
        int(torch.argmax(direct_capture["logits"]).item()),
        int(torch.argmax(target_capture["logits"]).item()),
    ]
    batch_top_ids = [int(value) for value in torch.argmax(batch_logits, dim=-1).tolist()]
    batch_equivalent = bool(
        batch_max_abs <= float(config["intervention"]["clean_batch_logit_atol"])
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
        "clean_batch_equivalent_at_registered_tolerance": batch_equivalent,
        "clean_batch_top_ids_equal": single_top_ids == batch_top_ids,
        "clean_single_top_ids": single_top_ids,
        "clean_batched_top_ids": batch_top_ids,
        "scientific_patch_batch_size": 1,
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


def _answer_contract(model, row: dict[str, Any], *, kind: str) -> dict[str, Any]:
    if kind == "direct":
        return {
            "source_id": model.concept_token_id(row["source"]),
            "target_id": model.concept_token_id(row["target"]),
            "wrong_id": model.concept_token_id(row["wrong"]),
            "parse_ids": {model.concept_token_id(concept) for concept in CONCEPTS},
        }
    if kind == "consequence":
        return {
            "source_id": model.bare_token_id(row["source_digit"]),
            "target_id": model.bare_token_id(row["target_digit"]),
            "wrong_id": model.bare_token_id(row["wrong_digit"]),
            "parse_ids": {model.bare_token_id(digit) for digit in DIGITS},
        }
    raise ValueError(kind)


def _scored_row(
    model,
    item: dict[str, Any],
    *,
    split: str,
    kind: str,
    condition: str,
    band: tuple[int, ...],
    score: dict[str, Any],
) -> dict[str, Any]:
    contract = _answer_contract(model, item, kind=kind)
    logits = score["logits"]
    top_id = int(score.get("top_id", int(__import__("torch").argmax(logits).item())))
    delta_norms = {
        str(layer): float(delta.float().norm()) for layer, delta in score.get("deltas", {}).items()
    }
    return {
        "item_id": item["item_id"],
        "split": split,
        "prompt_kind": kind,
        "condition": condition,
        "band": list(band),
        "source": item["source"],
        "target": item["target"],
        "wrong": item["wrong"],
        "source_answer": item["source"] if kind == "direct" else item["source_digit"],
        "target_answer": item["target"] if kind == "direct" else item["target_digit"],
        "wrong_answer": item["wrong"] if kind == "direct" else item["wrong_digit"],
        "source_id": contract["source_id"],
        "target_id": contract["target_id"],
        "wrong_id": contract["wrong_id"],
        "top_id": top_id,
        "top_text": model.tokenizer.decode([top_id]),
        "source_correct": top_id == contract["source_id"],
        "target_selected": top_id == contract["target_id"],
        "wrong_selected": top_id == contract["wrong_id"],
        "parsed": top_id in contract["parse_ids"],
        "target_minus_source_logit": float(
            logits[contract["target_id"]] - logits[contract["source_id"]]
        ),
        "wrong_minus_source_logit": float(
            logits[contract["wrong_id"]] - logits[contract["source_id"]]
        ),
        "delta_norms": delta_norms,
        "total_delta_norm": float(sum(value * value for value in delta_norms.values()) ** 0.5),
        "sequence_tokens": int(score["sequence_tokens"]),
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize empty rows")
    n = len(rows)
    return {
        "n": n,
        "source_accuracy": sum(bool(row["source_correct"]) for row in rows) / n,
        "target_rate": sum(bool(row["target_selected"]) for row in rows) / n,
        "wrong_rate": sum(bool(row["wrong_selected"]) for row in rows) / n,
        "parse_rate": sum(bool(row["parsed"]) for row in rows) / n,
        "mean_target_minus_source_logit": sum(
            float(row["target_minus_source_logit"]) for row in rows
        ) / n,
        "mean_total_delta_norm": sum(float(row["total_delta_norm"]) for row in rows) / n,
    }


def run_donor_gate(config: dict[str, Any]) -> dict[str, Any]:
    design = design_boundary_receipt(config)
    lens_receipt_path = RUNS_DIR / "lens_fit.json"
    if not lens_receipt_path.exists() or not read_json(lens_receipt_path).get("passed"):
        raise RuntimeError("eligible full-rank lens receipt is missing; run --stage fit-lens first")
    import time

    import torch

    from model_ops import FullActivationPatcher, QwenClampModel

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    model = QwenClampModel(config)
    rows = read_jsonl(DATA_DIR / "band_selection.jsonl")
    source_layers = tuple(int(value) for value in config["lens"]["source_layers"])
    bands = tuple(tuple(int(layer) for layer in band) for band in config["intervention"]["candidate_bands"])
    max_length = int(config["lens"]["max_sequence_tokens"])
    result_rows: list[dict[str, Any]] = []
    causal_differences: list[float] = []
    position_checks = 0

    for item in rows:
        prepared = {
            (kind, selected): _prepare_item(
                model, item, kind=kind, selected=selected, max_length=max_length
            )
            for kind in ("direct", "consequence")
            for selected in (item["source"], item["target"], item["wrong"])
        }
        positions = {value["position"] for value in prepared.values()}
        if len(positions) != 1:
            raise RuntimeError(f"selected-token positions disagree for {item['item_id']}: {positions}")
        for kind in ("direct", "consequence"):
            lengths = {
                prepared[(kind, selected)]["sequence_tokens"]
                for selected in (item["source"], item["target"], item["wrong"])
            }
            if len(lengths) != 1:
                raise RuntimeError(f"source/donor lengths disagree for {item['item_id']}/{kind}")
        position_checks += 1
        captures = {
            key: model.capture(value, layers=source_layers) for key, value in prepared.items()
        }
        for selected in (item["source"], item["target"], item["wrong"]):
            for layer in source_layers:
                causal_differences.append(float(
                    (
                        captures[("direct", selected)]["activations"][layer]
                        - captures[("consequence", selected)]["activations"][layer]
                    ).abs().max()
                ))

        for kind in ("direct", "consequence"):
            source_prepared = prepared[(kind, item["source"])]
            baseline = captures[(kind, item["source"])]
            result_rows.append(_scored_row(
                model,
                item,
                split="band_selection",
                kind=kind,
                condition="baseline",
                band=(),
                score=baseline,
            ))
            for band in bands:
                for condition, selected in (
                    ("full_target_donor", item["target"]),
                    ("full_wrong_donor", item["wrong"]),
                ):
                    desired = {
                        layer: captures[(kind, selected)]["activations"][layer]
                        for layer in band
                    }
                    patcher = FullActivationPatcher(
                        model.layers, source_prepared["position"], desired
                    )
                    scored = model.score(source_prepared, patcher=patcher)
                    result_rows.append(_scored_row(
                        model,
                        item,
                        split="band_selection",
                        kind=kind,
                        condition=condition,
                        band=band,
                        score=scored,
                    ))

    def subset(*, kind: str, condition: str, band: tuple[int, ...]) -> list[dict[str, Any]]:
        return [
            row for row in result_rows
            if row["prompt_kind"] == kind
            and row["condition"] == condition
            and row["band"] == list(band)
        ]

    baseline = {
        kind: _summary(subset(kind=kind, condition="baseline", band=()))
        for kind in ("direct", "consequence")
    }
    gates = config["gates"]
    clean_pass = all(
        baseline[kind]["source_accuracy"] >= float(gates["clean_accuracy_min"])
        and baseline[kind]["parse_rate"] >= float(gates["clean_parse_rate_min"])
        for kind in ("direct", "consequence")
    )
    candidates = []
    selected_band: tuple[int, ...] | None = None
    for band in bands:
        target = {
            kind: _summary(subset(kind=kind, condition="full_target_donor", band=band))
            for kind in ("direct", "consequence")
        }
        wrong = {
            kind: _summary(subset(kind=kind, condition="full_wrong_donor", band=band))
            for kind in ("direct", "consequence")
        }
        candidate_pass = bool(
            clean_pass
            and target["direct"]["target_rate"] >= float(gates["donor_direct_target_rate_min"])
            and target["consequence"]["target_rate"] >= float(gates["donor_consequence_target_rate_min"])
            and target["consequence"]["target_rate"] - wrong["consequence"]["target_rate"]
            >= float(gates["donor_target_minus_wrong_min"])
            and target["direct"]["parse_rate"] >= float(gates["clean_parse_rate_min"])
            and target["consequence"]["parse_rate"] >= float(gates["clean_parse_rate_min"])
        )
        candidates.append({
            "band": list(band),
            "passed": candidate_pass,
            "target_donor": target,
            "wrong_donor": wrong,
            "consequence_target_minus_wrong_target_rate": (
                target["consequence"]["target_rate"] - wrong["consequence"]["target_rate"]
            ),
        })
        if candidate_pass and selected_band is None:
            selected_band = band

    causal_max_abs = max(causal_differences, default=float("inf"))
    causal_pass = causal_max_abs <= float(config["intervention"]["causal_activation_atol"])
    passed = bool(design["passed"] and clean_pass and causal_pass and selected_band is not None)
    if not causal_pass:
        decision = "INVALID_CONTROL"
    elif passed:
        decision = "DONOR_GATE_PASS"
    else:
        decision = "NO_CAUSAL_SITE"
    result = {
        "schema_version": 1,
        "stage": "donor_gate",
        "passed": passed,
        "decision": decision,
        "scientific_result": True,
        "baseline": baseline,
        "clean_pass": clean_pass,
        "causal_activation_max_abs": causal_max_abs,
        "causal_invariance_pass": causal_pass,
        "position_contract_items": position_checks,
        "candidates": candidates,
        "selected_band": list(selected_band) if selected_band is not None else None,
        "selection_rule": "earliest_registered_passing_full_activation_donor_band",
        "j_outcomes_observed": False,
        "counts": {"items": len(rows), "rows": len(result_rows)},
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    write_jsonl(RUNS_DIR / "donor_gate_rows.jsonl", result_rows)
    write_json(RUNS_DIR / "donor_gate.json", result)
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
    if args.stage == "donor-gate":
        run_donor_gate(config)
        return 0
    design_boundary_receipt(config)
    unavailable(args.stage)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

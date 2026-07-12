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
    coordinate_score = model.score(source_direct, patcher=coordinate_patcher)
    random_smoke = _run_norm_matched_random(
        model,
        source_direct,
        directions={16: lens.directions[16]},
        reference_deltas=coordinate_score["deltas"],
        seed=int(config["seeds"]["controls"]),
        rtol=1e-5,
        tolerance=float(config["intervention"]["norm_match_relative_tolerance"]),
    )
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
        and random_smoke["passed"]
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
        "norm_matched_random_smoke": {
            "passed": random_smoke["passed"],
            "attempts": random_smoke["attempts"],
            "max_relative_error": max(random_smoke["relative_errors"].values()),
            "max_requested_span_projection_fraction": max(
                random_smoke["requested_projection_fractions"].values()
            ),
            "max_actual_span_projection_fraction": max(
                random_smoke["actual_projection_fractions"].values()
            ),
        },
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


def _stable_seed(base: int, *parts: str) -> int:
    payload = "\0".join((str(base), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") % (2**31)


def _run_norm_matched_random(
    model,
    prepared: dict[str, Any],
    *,
    directions: dict[int, Any],
    reference_deltas: dict[int, Any],
    seed: int,
    rtol: float,
    tolerance: float,
    max_attempts: int = 24,
) -> dict[str, Any]:
    import torch

    from coordinates import (
        orthogonal_norm_matched,
        span_projection_fraction,
    )
    from model_ops import NormMatchedDeltaPatcher

    target_norms = {
        layer: float(reference_deltas[layer].float().norm()) for layer in directions
    }
    requested = {}
    requested_projection = {}
    for layer in sorted(directions):
        candidates = []
        for attempt in range(1, max_attempts + 1):
            generator = torch.Generator().manual_seed(
                _stable_seed(seed, str(attempt), str(layer))
            )
            candidates.append(orthogonal_norm_matched(
                reference_deltas[layer].float(),
                directions[layer].float(),
                generator=generator,
                rtol=rtol,
            ))
        requested[layer] = torch.cat(candidates, dim=0)
        requested_projection[layer] = float(span_projection_fraction(
            requested[layer], directions[layer], rtol=rtol
        ).max())
    patcher = NormMatchedDeltaPatcher(
        model.layers,
        prepared["position"],
        requested,
        target_norms,
        search_steps=64,
    )
    score = model.score(prepared, patcher=patcher)
    actual_projection = {
        layer: float(span_projection_fraction(
            score["deltas"][layer].float(), directions[layer], rtol=rtol
        ).max())
        for layer in directions
    }
    return {
        "passed": max(patcher.relative_errors.values(), default=0.0) <= tolerance,
        "score": score,
        "attempts": max_attempts,
        "relative_errors": dict(patcher.relative_errors),
        "requested_projection_fractions": requested_projection,
        "actual_projection_fractions": actual_projection,
        "match_scales": dict(patcher.scales),
        "chosen_candidate_indices": dict(patcher.chosen_indices),
        "binary_search_steps": patcher.search_steps,
    }


def run_confirmation(config: dict[str, Any]) -> dict[str, Any]:
    design = design_boundary_receipt(config)
    donor_path = RUNS_DIR / "donor_gate.json"
    if not donor_path.exists():
        raise RuntimeError("donor gate receipt is missing; run --stage donor-gate first")
    donor_gate = read_json(donor_path)
    if not donor_gate.get("passed") or donor_gate.get("decision") != "DONOR_GATE_PASS":
        raise RuntimeError("confirmation is ineligible because the donor gate did not pass")
    lens_path = RUNS_DIR / "context_lens.pt"
    lens_receipt = read_json(RUNS_DIR / "lens_fit.json")
    if not lens_receipt.get("passed") or sha256_file(lens_path) != lens_receipt["artifact"]["sha256"]:
        raise RuntimeError("context lens is missing or does not match its eligible receipt")
    import time

    import torch

    from coordinates import read_coordinates
    from model_ops import ContextLens, CoordinateClampPatcher, FullActivationPatcher, QwenClampModel
    from stats import paired_bootstrap_mean_ci

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    model = QwenClampModel(config)
    lens = ContextLens.load(str(lens_path))
    if lens.concepts != tuple(CONCEPTS):
        raise RuntimeError("loaded lens concept order disagrees with frozen dictionary")
    rows = read_jsonl(DATA_DIR / "confirmation.jsonl")
    source_layers = tuple(int(value) for value in config["lens"]["source_layers"])
    band = tuple(int(value) for value in donor_gate["selected_band"])
    if list(band) not in config["intervention"]["candidate_bands"]:
        raise RuntimeError("stored donor band is not a registered candidate")
    rtol = float(config["lens"]["pseudoinverse_rtol"])
    tolerance = float(config["intervention"]["norm_match_relative_tolerance"])
    max_length = int(config["lens"]["max_sequence_tokens"])
    concept_index = {concept: index for index, concept in enumerate(lens.concepts)}
    j_directions = {layer: lens.directions[layer] for layer in band}
    logit_dictionary = model.lm_head.weight[list(lens.token_ids)].float().T.detach().cpu()
    logit_directions = {layer: logit_dictionary for layer in band}
    result_rows: list[dict[str, Any]] = []
    causal_differences: list[float] = []
    norm_audits = []

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
            raise RuntimeError(f"confirmation position contract failed: {item['item_id']}")
        for kind in ("direct", "consequence"):
            lengths = {
                prepared[(kind, selected)]["sequence_tokens"]
                for selected in (item["source"], item["target"], item["wrong"])
            }
            if len(lengths) != 1:
                raise RuntimeError(f"confirmation length contract failed: {item['item_id']}/{kind}")
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
            source_capture = captures[(kind, item["source"])]
            target_capture = captures[(kind, item["target"])]
            wrong_capture = captures[(kind, item["wrong"])]
            result_rows.append(_scored_row(
                model,
                item,
                split="confirmation",
                kind=kind,
                condition="baseline",
                band=(),
                score=source_capture,
            ))

            full_patcher = FullActivationPatcher(
                model.layers,
                source_prepared["position"],
                {layer: target_capture["activations"][layer] for layer in band},
            )
            result_rows.append(_scored_row(
                model,
                item,
                split="confirmation",
                kind=kind,
                condition="full_target_donor",
                band=band,
                score=model.score(source_prepared, patcher=full_patcher),
            ))

            desired_j = model.donor_coordinates(
                target_capture["activations"], j_directions, rtol=rtol
            )
            j_patcher = CoordinateClampPatcher(
                model.layers,
                source_prepared["position"],
                j_directions,
                desired_j,
                rtol=rtol,
            )
            j_score = model.score(source_prepared, patcher=j_patcher)
            result_rows.append(_scored_row(
                model,
                item,
                split="confirmation",
                kind=kind,
                condition="j_all24",
                band=band,
                score=j_score,
            ))

            random_audit = _run_norm_matched_random(
                model,
                source_prepared,
                directions=j_directions,
                reference_deltas=j_score["deltas"],
                seed=_stable_seed(int(config["seeds"]["controls"]), item["item_id"], kind),
                rtol=rtol,
                tolerance=tolerance,
            )
            random_row = _scored_row(
                model,
                item,
                split="confirmation",
                kind=kind,
                condition="random_norm_orthogonal",
                band=band,
                score=random_audit["score"],
            )
            random_row.update({
                "norm_match_passed": random_audit["passed"],
                "norm_match_attempts": random_audit["attempts"],
                "norm_match_max_relative_error": max(
                    random_audit["relative_errors"].values(), default=0.0
                ),
                "requested_span_projection_fraction_max": max(
                    random_audit["requested_projection_fractions"].values(), default=0.0
                ),
                "actual_span_projection_fraction_max": max(
                    random_audit["actual_projection_fractions"].values(), default=0.0
                ),
            })
            result_rows.append(random_row)
            norm_audits.append(random_row)

            desired_wrong = model.donor_coordinates(
                wrong_capture["activations"], j_directions, rtol=rtol
            )
            wrong_patcher = CoordinateClampPatcher(
                model.layers,
                source_prepared["position"],
                j_directions,
                desired_wrong,
                rtol=rtol,
            )
            result_rows.append(_scored_row(
                model,
                item,
                split="confirmation",
                kind=kind,
                condition="j_wrong_donor",
                band=band,
                score=model.score(source_prepared, patcher=wrong_patcher),
            ))

            pair_indices = [concept_index[item["source"]], concept_index[item["target"]]]
            pair_directions = {
                layer: lens.directions[layer][:, pair_indices] for layer in band
            }
            pair_desired = {
                layer: read_coordinates(
                    target_capture["activations"][layer].reshape(1, -1),
                    pair_directions[layer],
                    rtol=rtol,
                )[0]
                for layer in band
            }
            pair_patcher = CoordinateClampPatcher(
                model.layers,
                source_prepared["position"],
                pair_directions,
                pair_desired,
                rtol=rtol,
            )
            result_rows.append(_scored_row(
                model,
                item,
                split="confirmation",
                kind=kind,
                condition="j_pair",
                band=band,
                score=model.score(source_prepared, patcher=pair_patcher),
            ))

            desired_logit = model.donor_coordinates(
                target_capture["activations"], logit_directions, rtol=rtol
            )
            logit_patcher = CoordinateClampPatcher(
                model.layers,
                source_prepared["position"],
                logit_directions,
                desired_logit,
                rtol=rtol,
            )
            result_rows.append(_scored_row(
                model,
                item,
                split="confirmation",
                kind=kind,
                condition="logit_lens_all24",
                band=band,
                score=model.score(source_prepared, patcher=logit_patcher),
            ))

    def subset(*, kind: str, condition: str) -> list[dict[str, Any]]:
        return [
            row for row in result_rows
            if row["prompt_kind"] == kind and row["condition"] == condition
        ]

    conditions = (
        "baseline",
        "full_target_donor",
        "j_all24",
        "random_norm_orthogonal",
        "j_wrong_donor",
        "j_pair",
        "logit_lens_all24",
    )
    summaries = {
        condition: {
            kind: _summary(subset(kind=kind, condition=condition))
            for kind in ("direct", "consequence")
        }
        for condition in conditions
    }
    baseline = summaries["baseline"]
    primary = summaries["j_all24"]
    random_control = summaries["random_norm_orthogonal"]
    wrong_control = summaries["j_wrong_donor"]
    gates = config["gates"]
    clean_pass = all(
        baseline[kind]["source_accuracy"] >= float(gates["clean_accuracy_min"])
        and baseline[kind]["parse_rate"] >= float(gates["clean_parse_rate_min"])
        for kind in ("direct", "consequence")
    )
    donor_pass = bool(
        summaries["full_target_donor"]["direct"]["target_rate"]
        >= float(gates["donor_direct_target_rate_min"])
        and summaries["full_target_donor"]["consequence"]["target_rate"]
        >= float(gates["donor_consequence_target_rate_min"])
    )
    norm_pass = all(bool(row["norm_match_passed"]) for row in norm_audits)
    causal_max_abs = max(causal_differences, default=float("inf"))
    causal_pass = causal_max_abs <= float(config["intervention"]["causal_activation_atol"])
    j_by_item = {row["item_id"]: float(row["target_selected"]) for row in subset(
        kind="consequence", condition="j_all24"
    )}
    random_by_item = {row["item_id"]: float(row["target_selected"]) for row in subset(
        kind="consequence", condition="random_norm_orthogonal"
    )}
    if set(j_by_item) != set(random_by_item):
        raise RuntimeError("paired J/random confirmation item sets disagree")
    differences = [j_by_item[item_id] - random_by_item[item_id] for item_id in sorted(j_by_item)]
    bootstrap = paired_bootstrap_mean_ci(
        differences,
        resamples=int(gates["bootstrap_resamples"]),
        seed=int(config["seeds"]["bootstrap"]),
    )
    direct_shift = primary["direct"]["target_rate"] - baseline["direct"]["target_rate"]
    consequence_shift = (
        primary["consequence"]["target_rate"] - baseline["consequence"]["target_rate"]
    )
    j_minus_random = (
        primary["consequence"]["target_rate"]
        - random_control["consequence"]["target_rate"]
    )
    j_minus_wrong_target = (
        primary["consequence"]["target_rate"] - wrong_control["consequence"]["target_rate"]
    )
    wrong_own_shift = (
        wrong_control["consequence"]["wrong_rate"] - baseline["consequence"]["wrong_rate"]
    )
    parse_drop = baseline["consequence"]["parse_rate"] - primary["consequence"]["parse_rate"]
    primary_pass = bool(
        clean_pass
        and donor_pass
        and norm_pass
        and causal_pass
        and direct_shift >= float(gates["j_direct_shift_min"])
        and consequence_shift >= float(gates["j_consequence_shift_min"])
        and j_minus_random >= float(gates["j_minus_random_min"])
        and j_minus_wrong_target >= float(gates["j_minus_wrong_target_min"])
        and wrong_own_shift >= float(gates["wrong_own_digit_shift_min"])
        and parse_drop <= float(gates["max_parse_rate_drop"])
        and bootstrap["lower"] > float(gates["bootstrap_lower_bound_min"])
    )
    if not norm_pass or not causal_pass:
        decision = "INVALID_CONTROL"
    elif not donor_pass:
        decision = "INVALID_CONTROL"
    elif primary_pass:
        decision = "J_TRANSPORT"
    elif direct_shift >= float(gates["j_direct_shift_min"]):
        decision = "DIRECT_ONLY"
    else:
        decision = "DONOR_ONLY"
    result = {
        "schema_version": 1,
        "stage": "confirmation",
        "passed": primary_pass,
        "decision": decision,
        "scientific_result": True,
        "band": list(band),
        "confirmation_n": len(rows),
        "summaries": summaries,
        "gate_metrics": {
            "clean_pass": clean_pass,
            "donor_pass": donor_pass,
            "causal_invariance_pass": causal_pass,
            "norm_match_pass": norm_pass,
            "direct_target_shift": direct_shift,
            "consequence_target_shift": consequence_shift,
            "consequence_j_minus_random": j_minus_random,
            "consequence_j_minus_wrong_target": j_minus_wrong_target,
            "wrong_donor_own_digit_shift": wrong_own_shift,
            "consequence_parse_drop": parse_drop,
            "paired_bootstrap_j_minus_random": bootstrap,
        },
        "control_audit": {
            "norm_tolerance": tolerance,
            "max_norm_relative_error": max(
                (float(row["norm_match_max_relative_error"]) for row in norm_audits),
                default=0.0,
            ),
            "max_norm_match_attempts": max(
                (int(row["norm_match_attempts"]) for row in norm_audits), default=0
            ),
            "max_requested_span_projection_fraction": max(
                (float(row["requested_span_projection_fraction_max"]) for row in norm_audits),
                default=0.0,
            ),
            "max_actual_span_projection_fraction": max(
                (float(row["actual_span_projection_fraction_max"]) for row in norm_audits),
                default=0.0,
            ),
            "causal_activation_max_abs": causal_max_abs,
            "target_digit_gradient_used": False,
            "patch_batch_size": 1,
            "use_cache": False,
        },
        "counts": {"items": len(rows), "rows": len(result_rows)},
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    write_jsonl(RUNS_DIR / "confirmation_rows.jsonl", result_rows)
    write_json(RUNS_DIR / "confirmation.json", result)
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
    if args.stage == "confirmation":
        run_confirmation(config)
        return 0
    design_boundary_receipt(config)
    unavailable(args.stage)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

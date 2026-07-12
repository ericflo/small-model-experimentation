#!/usr/bin/env python3
"""Stage-gated quantization-aware J-transport replication harness."""

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

from io_utils import read_json, read_jsonl, sha256_file, write_json, write_jsonl  # noqa: E402
from task_data import (  # noqa: E402
    CONCEPTS,
    DIGITS,
    consequence_prompt,
    direct_prompt,
    fingerprint,
    generate_replication_splits,
)


CONFIG_PATH = EXP / "configs" / "default.yaml"
DATA_DIR = EXP / "data" / "procedural"
RUNS_DIR = EXP / "runs"
CALIBRATION_UNLOCK = {
    "commit": "2bd6376c28283546a111e54ec2dd2e92a0fd6a64",
    "summary_sha256": "58ac9086b86efcae78180476e5631024f1e6c3afe08cc3f562ba5d35ea7ba22b",
    "rows_sha256": "be7c73edfff88b79bd9f4f115c5086cf56bcf5927f77ee95a92135244f3ea27d",
}


def load_config() -> dict[str, Any]:
    value = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
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
    commit = str(boundary["commit"])
    head = _git(["rev-parse", "HEAD"]).stdout.strip()
    ancestor = _git(["merge-base", "--is-ancestor", commit, head], check=False).returncode == 0
    paths = {
        "preregistration": (
            "experiments/qwen35_4b_jacobian_transport_control_replication/"
            "reports/preregistration.md"
        ),
        "readme": "experiments/qwen35_4b_jacobian_transport_control_replication/README.md",
    }
    observed = {
        name: hashlib.sha256(_git(["show", f"{commit}:{path}"]).stdout.encode()).hexdigest()
        for name, path in paths.items()
    }
    expected = {
        "preregistration": str(boundary["preregistration_sha256"]),
        "readme": str(boundary["readme_sha256"]),
    }
    lens_path = EXP / config["lens"]["path"]
    lens_observed = sha256_file(lens_path)
    passed = bool(
        ancestor and observed == expected and lens_observed == str(config["lens"]["sha256"])
    )
    receipt = {
        "schema_version": 1,
        "passed": passed,
        "scientific_result": False,
        "design_commit": commit,
        "head": head,
        "design_is_ancestor": ancestor,
        "observed_sha256": observed,
        "expected_sha256": expected,
        "lens_observed_sha256": lens_observed,
        "lens_expected_sha256": str(config["lens"]["sha256"]),
    }
    write_json(RUNS_DIR / "design_boundary_receipt.json", receipt)
    if not passed:
        raise RuntimeError(f"immutable design boundary failed: {receipt}")
    return receipt


def _parent_fingerprints() -> set[str]:
    fingerprints: set[str] = set()
    parents = (
        ROOT / "experiments" / "qwen35_4b_context_local_jacobian_clamp" / "data" / "procedural",
        ROOT / "experiments" / "qwen35_4b_jacobian_value_transport" / "data" / "procedural",
    )
    for directory in parents:
        for path in sorted(directory.glob("*.jsonl")):
            for row in read_jsonl(path):
                if {"mapping", "source", "target", "wrong"}.issubset(row):
                    fingerprints.add(fingerprint(row))
    return fingerprints


def run_smoke(config: dict[str, Any]) -> dict[str, Any]:
    if config["model"]["id"] != "Qwen/Qwen3.5-4B":
        raise RuntimeError("only Qwen/Qwen3.5-4B is permitted")
    lens_path = EXP / config["lens"]["path"]
    lens_hash = sha256_file(lens_path)
    if lens_hash != config["lens"]["sha256"]:
        raise RuntimeError("frozen parent lens hash mismatch")
    splits = generate_replication_splits(config)
    parent = _parent_fingerprints()
    overlap = {
        name: sorted(fingerprint(row) for row in rows if fingerprint(row) in parent)
        for name, rows in splits.items()
    }
    if any(overlap.values()):
        raise RuntimeError(f"fresh replication rows overlap parent data: {overlap}")
    paths = {}
    for name, rows in splits.items():
        path = DATA_DIR / f"{name}.jsonl"
        write_jsonl(path, rows)
        paths[name] = path
    manifest = {
        "schema_version": 1,
        "model_id": config["model"]["id"],
        "model_revision": config["model"]["revision"],
        "frozen_lens_sha256": lens_hash,
        "parent_fingerprint_count": len(parent),
        "parent_overlap_count": sum(len(values) for values in overlap.values()),
        "splits": {
            name: {
                "items": len(rows),
                "path": str(paths[name].relative_to(EXP)),
                "sha256": sha256_file(paths[name]),
                "unique_fingerprints": len({fingerprint(row) for row in rows}),
                "source_counts": {
                    concept: sum(row["source"] == concept for row in rows)
                    for concept in CONCEPTS
                },
            }
            for name, rows in splits.items()
        },
        "scientific_result": False,
    }
    write_json(DATA_DIR / "manifest.json", manifest)
    receipt = {
        "schema_version": 1,
        "stage": "cpu_smoke",
        "passed": True,
        "scientific_result": False,
        "lens_sha256": lens_hash,
        "parent_overlap_count": 0,
        "split_sizes": {name: len(rows) for name, rows in splits.items()},
        "band": config["intervention"]["band"],
        "random_arms": config["intervention"]["random_arms"],
        "norm_tolerance": config["intervention"]["norm_relative_tolerance"],
        "projection_tolerance": config["intervention"]["realized_span_projection_max"],
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


def _stable_seed(base: int, *parts: str) -> int:
    payload = "\0".join((str(base), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") % (2**31)


def _make_control_bases(
    reference_delta,
    directions,
    *,
    base_seed: int,
    item_id: str,
    kind: str,
    arm: str,
    layer: int,
    draws: int,
    rtol: float,
):
    import torch

    from coordinates import orthogonal_norm_matched

    values = []
    for draw in range(draws):
        generator = torch.Generator().manual_seed(
            _stable_seed(base_seed, item_id, kind, arm, str(layer), str(draw))
        )
        values.append(orthogonal_norm_matched(
            reference_delta.float(), directions.float(), generator=generator, rtol=rtol
        ))
    return torch.cat(values, dim=0)


def _control_patcher(
    model,
    prepared,
    *,
    config: dict[str, Any],
    item_id: str,
    kind: str,
    arm: str,
    directions: dict[int, Any],
    reference_deltas: dict[int, Any],
):
    from model_ops import QuantizationAwareOrthogonalPatcher

    intervention = config["intervention"]
    rtol = float(config["lens"]["pseudoinverse_rtol"])
    bases = {
        layer: _make_control_bases(
            reference_deltas[layer],
            directions[layer],
            base_seed=int(config["seeds"][arm]),
            item_id=item_id,
            kind=kind,
            arm=arm,
            layer=layer,
            draws=int(intervention["candidate_draws"]),
            rtol=rtol,
        )
        for layer in directions
    }
    return QuantizationAwareOrthogonalPatcher(
        model.layers,
        prepared["position"],
        bases,
        directions,
        {
            layer: float(reference_deltas[layer].float().norm())
            for layer in directions
        },
        rtol=rtol,
        norm_tolerance=float(intervention["norm_relative_tolerance"]),
        projection_tolerance=float(intervention["realized_span_projection_max"]),
        correction_iterations=int(intervention["correction_iterations"]),
        correction_damping=float(intervention["correction_damping"]),
        binary_search_steps=int(intervention["binary_search_steps"]),
    )


def _numeric_rows(patcher, *, item_id: str, kind: str, arm: str) -> list[dict[str, Any]]:
    return [
        {
            "item_id": item_id,
            "prompt_kind": kind,
            "arm": arm,
            "layer": layer,
            "passed": bool(patcher.passed_by_layer[layer]),
            "j_delta_norm": float(patcher.target_norms_by_layer[layer]),
            "control_delta_norm": float(patcher.deltas[layer].float().norm()),
            "norm_relative_error": float(patcher.norm_errors[layer]),
            "realized_span_projection_fraction": float(
                patcher.projection_fractions[layer]
            ),
            "chosen_candidate_index": int(patcher.chosen_indices[layer]),
            "correction_iterations": int(patcher.iterations_used[layer]),
            "lattice_pair_steps": int(patcher.lattice_pair_steps[layer]),
        }
        for layer in sorted(patcher.deltas)
    ]


def _load_model_and_lens(config: dict[str, Any]):
    from model_ops import ContextLens, QwenClampModel

    model = QwenClampModel(config)
    lens = ContextLens.load(str(EXP / config["lens"]["path"]))
    if lens.concepts != tuple(CONCEPTS):
        raise RuntimeError("frozen lens concept order mismatch")
    return model, lens


def run_model_smoke(config: dict[str, Any]) -> dict[str, Any]:
    design = design_boundary_receipt(config)
    import time

    import torch
    import transformers

    from coordinates import dictionary_stats
    from model_ops import CoordinateClampPatcher

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    model, lens = _load_model_and_lens(config)
    band = tuple(int(layer) for layer in config["intervention"]["band"])
    directions = {layer: lens.directions[layer] for layer in band}
    rtol = float(config["lens"]["pseudoinverse_rtol"])
    stats = {layer: dictionary_stats(directions[layer], rtol=rtol) for layer in band}
    token_contract = {
        "concepts": {concept: model.concept_token_id(concept) for concept in CONCEPTS},
        "digits": {digit: model.bare_token_id(digit) for digit in DIGITS},
    }
    item = read_jsonl(DATA_DIR / "control_calibration.jsonl")[0]
    max_length = int(config["intervention"]["max_sequence_tokens"])
    prepared = {
        (kind, selected): _prepare_item(
            model, item, kind=kind, selected=selected, max_length=max_length
        )
        for kind in ("direct", "consequence")
        for selected in (item["source"], item["target"])
    }
    position_pass = len({value["position"] for value in prepared.values()}) == 1
    length_pass = all(
        len({
            prepared[(kind, selected)]["sequence_tokens"]
            for selected in (item["source"], item["target"])
        }) == 1
        for kind in ("direct", "consequence")
    )
    captures = {key: model.capture(value, layers=band) for key, value in prepared.items()}
    causal_max = max(
        float((
            captures[("direct", selected)]["activations"][layer]
            - captures[("consequence", selected)]["activations"][layer]
        ).abs().max())
        for selected in (item["source"], item["target"])
        for layer in band
    )
    numeric_rows = []
    for kind in ("direct", "consequence"):
        source_prepared = prepared[(kind, item["source"])]
        target_capture = captures[(kind, item["target"])]
        desired = model.donor_coordinates(target_capture["activations"], directions, rtol=rtol)
        j_patcher = CoordinateClampPatcher(
            model.layers, source_prepared["position"], directions, desired, rtol=rtol
        )
        j_score = model.score(source_prepared, patcher=j_patcher)
        reference = dict(j_score["deltas"])
        del j_score
        for arm in config["intervention"]["random_arms"]:
            patcher = _control_patcher(
                model,
                source_prepared,
                config=config,
                item_id=item["item_id"],
                kind=kind,
                arm=arm,
                directions=directions,
                reference_deltas=reference,
            )
            score = model.score(source_prepared, patcher=patcher)
            del score
            numeric_rows.extend(_numeric_rows(
                patcher, item_id=item["item_id"], kind=kind, arm=arm
            ))
    passed = bool(
        design["passed"]
        and model.n_layers == 32
        and model.d_model == 2560
        and all(stat.effective_rank == 24 for stat in stats.values())
        and position_pass
        and length_pass
        and causal_max <= float(config["intervention"]["causal_activation_atol"])
        and all(row["passed"] for row in numeric_rows)
    )
    result = {
        "schema_version": 1,
        "stage": "model_smoke",
        "passed": passed,
        "scientific_result": False,
        "outcomes_recorded": False,
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
        "lens_effective_ranks": {str(layer): stats[layer].effective_rank for layer in band},
        "causal_activation_max_abs": causal_max,
        "position_contract_pass": position_pass,
        "length_contract_pass": length_pass,
        "numeric_control_rows": numeric_rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(RUNS_DIR / "model_smoke" / "result.json", result)
    if not passed:
        raise RuntimeError(f"model smoke failed: {result}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def run_control_calibration(config: dict[str, Any]) -> dict[str, Any]:
    design = design_boundary_receipt(config)
    import time

    import torch

    from model_ops import CoordinateClampPatcher

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    model, lens = _load_model_and_lens(config)
    band = tuple(int(layer) for layer in config["intervention"]["band"])
    directions = {layer: lens.directions[layer] for layer in band}
    rtol = float(config["lens"]["pseudoinverse_rtol"])
    max_length = int(config["intervention"]["max_sequence_tokens"])
    items = read_jsonl(DATA_DIR / "control_calibration.jsonl")
    numeric_rows: list[dict[str, Any]] = []
    causal_differences: list[float] = []
    for item in items:
        prepared = {
            (kind, selected): _prepare_item(
                model, item, kind=kind, selected=selected, max_length=max_length
            )
            for kind in ("direct", "consequence")
            for selected in (item["source"], item["target"])
        }
        if len({value["position"] for value in prepared.values()}) != 1:
            raise RuntimeError(f"calibration position mismatch: {item['item_id']}")
        for kind in ("direct", "consequence"):
            if len({
                prepared[(kind, selected)]["sequence_tokens"]
                for selected in (item["source"], item["target"])
            }) != 1:
                raise RuntimeError(f"calibration length mismatch: {item['item_id']}/{kind}")
        captures = {key: model.capture(value, layers=band) for key, value in prepared.items()}
        for capture in captures.values():
            capture.pop("logits", None)
        for selected in (item["source"], item["target"]):
            for layer in band:
                causal_differences.append(float((
                    captures[("direct", selected)]["activations"][layer]
                    - captures[("consequence", selected)]["activations"][layer]
                ).abs().max()))
        for kind in ("direct", "consequence"):
            source_prepared = prepared[(kind, item["source"])]
            desired = model.donor_coordinates(
                captures[(kind, item["target"])]["activations"], directions, rtol=rtol
            )
            j_patcher = CoordinateClampPatcher(
                model.layers, source_prepared["position"], directions, desired, rtol=rtol
            )
            j_score = model.score(source_prepared, patcher=j_patcher)
            reference = dict(j_score["deltas"])
            del j_score
            for arm in config["intervention"]["random_arms"]:
                patcher = _control_patcher(
                    model,
                    source_prepared,
                    config=config,
                    item_id=item["item_id"],
                    kind=kind,
                    arm=arm,
                    directions=directions,
                    reference_deltas=reference,
                )
                score = model.score(source_prepared, patcher=patcher)
                del score
                numeric_rows.extend(_numeric_rows(
                    patcher, item_id=item["item_id"], kind=kind, arm=arm
                ))
    expected = (
        len(items)
        * 2
        * len(config["intervention"]["random_arms"])
        * len(band)
    )
    causal_max = max(causal_differences, default=float("inf"))
    passed = bool(
        design["passed"]
        and len(numeric_rows) == expected
        and all(row["passed"] for row in numeric_rows)
        and causal_max <= float(config["intervention"]["causal_activation_atol"])
    )
    result = {
        "schema_version": 1,
        "stage": "control_calibration",
        "passed": passed,
        "decision": "CONTROL_CALIBRATION_PASS" if passed else "CONTROL_UNREACHABLE",
        "scientific_result": False,
        "outcomes_recorded": False,
        "logits_recorded": False,
        "items": len(items),
        "numeric_rows": len(numeric_rows),
        "expected_numeric_rows": expected,
        "causal_activation_max_abs": causal_max,
        "max_norm_relative_error": max(
            (row["norm_relative_error"] for row in numeric_rows), default=float("inf")
        ),
        "max_realized_span_projection_fraction": max(
            (row["realized_span_projection_fraction"] for row in numeric_rows),
            default=float("inf"),
        ),
        "max_correction_iterations": max(
            (row["correction_iterations"] for row in numeric_rows), default=0
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    forbidden_exact = {
        "top_id", "top_text", "answer", "source_correct", "target_selected",
        "wrong_selected", "target_rate", "source_accuracy",
    }
    if any(
        key in forbidden_exact or "logit" in key
        for row in numeric_rows for key in row
    ):
        raise RuntimeError("numeric calibration artifact contains an outcome-like field")
    write_jsonl(RUNS_DIR / "control_calibration_rows.jsonl", numeric_rows)
    write_json(RUNS_DIR / "control_calibration.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def confirmation_unlock(config: dict[str, Any]) -> dict[str, Any]:
    """Require the exact committed, outcome-blind calibration firewall."""
    from confirmation import validate_calibration_contract

    summary_path = RUNS_DIR / "control_calibration.json"
    rows_path = RUNS_DIR / "control_calibration_rows.jsonl"
    if not summary_path.exists() or not rows_path.exists():
        raise RuntimeError("committed calibration artifacts are missing")
    head = _git(["rev-parse", "HEAD"]).stdout.strip()
    commit = str(CALIBRATION_UNLOCK["commit"])
    ancestor = (
        _git(["merge-base", "--is-ancestor", commit, head], check=False).returncode
        == 0
    )
    observed = {
        "summary_sha256": sha256_file(summary_path),
        "rows_sha256": sha256_file(rows_path),
    }
    expected = {
        "summary_sha256": str(CALIBRATION_UNLOCK["summary_sha256"]),
        "rows_sha256": str(CALIBRATION_UNLOCK["rows_sha256"]),
    }
    contract = validate_calibration_contract(
        read_json(summary_path), read_jsonl(rows_path), config
    )
    passed = bool(ancestor and observed == expected and contract["passed"])
    result = {
        "schema_version": 1,
        "stage": "confirmation_unlock",
        "passed": passed,
        "scientific_result": False,
        "calibration_commit": commit,
        "head": head,
        "calibration_is_ancestor": ancestor,
        "observed_sha256": observed,
        "expected_sha256": expected,
        "calibration_contract": contract,
    }
    write_json(RUNS_DIR / "confirmation_unlock.json", result)
    if not passed:
        raise RuntimeError(f"confirmation calibration firewall failed: {result}")
    return result


def run_confirmation(config: dict[str, Any]) -> dict[str, Any]:
    design = design_boundary_receipt(config)
    unlock = confirmation_unlock(config)
    import time

    import torch

    from confirmation import evaluate_confirmation, scored_row
    from coordinates import read_coordinates
    from model_ops import CoordinateClampPatcher, FullActivationPatcher

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    model, lens = _load_model_and_lens(config)
    items = read_jsonl(DATA_DIR / "confirmation.jsonl")
    expected_items = int(config["data"]["confirmation_items"])
    if len(items) != expected_items:
        raise RuntimeError("untouched confirmation cardinality changed")
    band = tuple(int(layer) for layer in config["intervention"]["band"])
    rtol = float(config["lens"]["pseudoinverse_rtol"])
    max_length = int(config["intervention"]["max_sequence_tokens"])
    concept_index = {concept: index for index, concept in enumerate(lens.concepts)}
    j_directions = {layer: lens.directions[layer] for layer in band}
    logit_dictionary = (
        model.lm_head.weight[list(lens.token_ids)].float().T.detach().cpu()
    )
    logit_directions = {layer: logit_dictionary for layer in band}
    result_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    causal_differences: list[float] = []
    position_contract_items = 0

    def record(
        item: dict[str, Any],
        *,
        kind: str,
        condition: str,
        score: dict[str, Any],
        row_band: tuple[int, ...],
    ) -> dict[str, Any]:
        row = scored_row(
            model,
            item,
            kind=kind,
            condition=condition,
            band=row_band,
            score=score,
            concepts=CONCEPTS,
            digits=DIGITS,
        )
        result_rows.append(row)
        return row

    for item in items:
        prepared = {
            (kind, selected): _prepare_item(
                model, item, kind=kind, selected=selected, max_length=max_length
            )
            for kind in ("direct", "consequence")
            for selected in (item["source"], item["target"], item["wrong"])
        }
        if len({value["position"] for value in prepared.values()}) != 1:
            raise RuntimeError(
                f"confirmation position contract failed: {item['item_id']}"
            )
        for kind in ("direct", "consequence"):
            lengths = {
                prepared[(kind, selected)]["sequence_tokens"]
                for selected in (item["source"], item["target"], item["wrong"])
            }
            if len(lengths) != 1:
                raise RuntimeError(
                    f"confirmation length contract failed: {item['item_id']}/{kind}"
                )
        position_contract_items += 1
        captures = {
            key: model.capture(value, layers=band) for key, value in prepared.items()
        }
        for selected in (item["source"], item["target"], item["wrong"]):
            for layer in band:
                causal_differences.append(float((
                    captures[("direct", selected)]["activations"][layer]
                    - captures[("consequence", selected)]["activations"][layer]
                ).abs().max()))

        for kind in ("direct", "consequence"):
            source_prepared = prepared[(kind, item["source"])]
            source_capture = captures[(kind, item["source"])]
            target_capture = captures[(kind, item["target"])]
            wrong_capture = captures[(kind, item["wrong"])]
            record(
                item,
                kind=kind,
                condition="baseline",
                score=source_capture,
                row_band=(),
            )

            full_patcher = FullActivationPatcher(
                model.layers,
                source_prepared["position"],
                {layer: target_capture["activations"][layer] for layer in band},
            )
            record(
                item,
                kind=kind,
                condition="full_target_donor",
                score=model.score(source_prepared, patcher=full_patcher),
                row_band=band,
            )

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
            record(
                item,
                kind=kind,
                condition="j_all24",
                score=j_score,
                row_band=band,
            )
            reference_deltas = dict(j_score["deltas"])

            for arm in config["intervention"]["random_arms"]:
                control_patcher = _control_patcher(
                    model,
                    source_prepared,
                    config=config,
                    item_id=item["item_id"],
                    kind=kind,
                    arm=arm,
                    directions=j_directions,
                    reference_deltas=reference_deltas,
                )
                control_score = model.score(
                    source_prepared, patcher=control_patcher
                )
                numeric = _numeric_rows(
                    control_patcher,
                    item_id=item["item_id"],
                    kind=kind,
                    arm=arm,
                )
                control_rows.extend(numeric)
                outcome = record(
                    item,
                    kind=kind,
                    condition=arm,
                    score=control_score,
                    row_band=band,
                )
                outcome.update({
                    "control_geometry_pass": all(row["passed"] for row in numeric),
                    "control_max_norm_relative_error": max(
                        row["norm_relative_error"] for row in numeric
                    ),
                    "control_max_realized_span_projection_fraction": max(
                        row["realized_span_projection_fraction"] for row in numeric
                    ),
                    "control_lattice_pair_steps": sum(
                        row["lattice_pair_steps"] for row in numeric
                    ),
                })

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
            record(
                item,
                kind=kind,
                condition="j_wrong_donor",
                score=model.score(source_prepared, patcher=wrong_patcher),
                row_band=band,
            )

            pair_indices = [
                concept_index[item["source"]], concept_index[item["target"]]
            ]
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
            record(
                item,
                kind=kind,
                condition="j_pair",
                score=model.score(source_prepared, patcher=pair_patcher),
                row_band=band,
            )

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
            record(
                item,
                kind=kind,
                condition="logit_lens_all24",
                score=model.score(source_prepared, patcher=logit_patcher),
                row_band=band,
            )

    causal_max_abs = max(causal_differences, default=float("inf"))
    evaluation = evaluate_confirmation(
        result_rows,
        control_rows,
        config,
        design_pass=bool(design["passed"]),
        calibration_pass=bool(unlock["passed"]),
        causal_max_abs=causal_max_abs,
    )
    result = {
        "schema_version": 1,
        "stage": "confirmation",
        "passed": bool(evaluation["passed"]),
        "decision": evaluation["decision"],
        "scientific_result": True,
        "oracle_mechanism_only": True,
        "band": list(band),
        "confirmation_n": len(items),
        "summaries": evaluation["summaries"],
        "gate_metrics": evaluation["gate_metrics"],
        "control_audit": {
            **evaluation["control_audit"],
            "norm_tolerance": float(
                config["intervention"]["norm_relative_tolerance"]
            ),
            "projection_tolerance": float(
                config["intervention"]["realized_span_projection_max"]
            ),
            "causal_activation_max_abs": causal_max_abs,
            "target_digit_gradient_used": False,
            "patch_batch_size": 1,
            "use_cache": False,
            "calibration_commit": unlock["calibration_commit"],
            "calibration_hashes": unlock["observed_sha256"],
        },
        "contracts": {
            "design_pass": bool(design["passed"]),
            "calibration_unlock_pass": bool(unlock["passed"]),
            "position_contract_items": position_contract_items,
            "model_id": config["model"]["id"],
            "model_revision": config["model"]["revision"],
            "backend": "transformers_bfloat16_sdpa_batch_one_no_cache",
        },
        "counts": {
            "items": len(items),
            "outcome_rows": len(result_rows),
            "numeric_control_rows": len(control_rows),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    write_jsonl(RUNS_DIR / "confirmation_rows.jsonl", result_rows)
    write_jsonl(RUNS_DIR / "confirmation_control_rows.jsonl", control_rows)
    write_json(RUNS_DIR / "confirmation.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def unavailable(stage: str) -> None:
    raise RuntimeError(f"stage {stage!r} is not implemented; refusing a placeholder result")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("smoke", "model-smoke", "control-calibration", "confirmation", "full"),
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
    if args.stage == "control-calibration":
        run_control_calibration(config)
        return 0
    if args.stage == "confirmation":
        run_confirmation(config)
        return 0
    design_boundary_receipt(config)
    unavailable(args.stage)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

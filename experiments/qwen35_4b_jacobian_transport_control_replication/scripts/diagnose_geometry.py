#!/usr/bin/env python3
"""Outcome-blind layer-8 feasibility diagnostics for the bf16 control lattice."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))
sys.path.insert(0, str(EXP / "scripts"))

from io_utils import read_jsonl  # noqa: E402
from model_ops import QuantizationAwareOrthogonalPatcher  # noqa: E402
from run import (  # noqa: E402
    DATA_DIR,
    _make_control_bases,
    _control_patcher,
    _prepare_item,
    load_config,
    _load_model_and_lens,
)


def main() -> None:
    config = load_config()
    model, lens = _load_model_and_lens(config)
    item = read_jsonl(DATA_DIR / "control_calibration.jsonl")[0]
    layer = 8
    rtol = float(config["lens"]["pseudoinverse_rtol"])
    direction = lens.directions[layer]
    max_length = int(config["intervention"]["max_sequence_tokens"])
    rows = []
    for kind in ("direct", "consequence"):
        source = _prepare_item(
            model, item, kind=kind, selected=item["source"], max_length=max_length
        )
        target = _prepare_item(
            model, item, kind=kind, selected=item["target"], max_length=max_length
        )
        band = tuple(int(value) for value in config["intervention"]["band"])
        directions = {value: lens.directions[value] for value in band}
        target_band = model.capture(target, layers=band)["activations"]
        desired_band = model.donor_coordinates(target_band, directions, rtol=rtol)
        from model_ops import CoordinateClampPatcher

        j_patcher = CoordinateClampPatcher(
            model.layers, source["position"], directions, desired_band, rtol=rtol
        )
        j_score = model.score(source, patcher=j_patcher)
        references = dict(j_score["deltas"])
        del j_score
        reference = references[layer]
        target_norm = float(reference.norm())
        for arm in config["intervention"]["random_arms"]:
            bases = _make_control_bases(
                references[layer],
                direction,
                base_seed=int(config["seeds"][arm]),
                item_id=item["item_id"],
                kind=kind,
                arm=arm,
                layer=layer,
                draws=int(config["intervention"]["candidate_draws"]),
                rtol=rtol,
            ).cuda()
            live_patcher = _control_patcher(
                model,
                source,
                config=config,
                item_id=item["item_id"],
                kind=kind,
                arm=arm,
                directions=directions,
                reference_deltas=references,
            )
            live_score = model.score(source, patcher=live_patcher)
            del live_score
            live_current = live_patcher.input_activations[layer].to(
                device="cuda", dtype=torch.bfloat16
            )
            angles = torch.linspace(0.0, math.pi / 2.0, 65, device="cuda")
            mixtures = []
            for offset in (1, 3, 7, 13):
                neighbor = bases.roll(offset, dims=0)
                mixed = (
                    torch.cos(angles)[:, None, None] * bases[None, :, :]
                    + torch.sin(angles)[:, None, None] * neighbor[None, :, :]
                ).reshape(-1, bases.shape[1])
                mixed *= target_norm / mixed.norm(dim=-1, keepdim=True)
                mixtures.append(mixed)
            candidates = torch.cat(mixtures)
            dummy = torch.nn.ModuleList([torch.nn.Identity() for _ in range(layer + 1)])
            patcher = QuantizationAwareOrthogonalPatcher(
                dummy,
                0,
                {layer: bases.cpu()},
                {layer: direction},
                {layer: target_norm},
                rtol=rtol,
                norm_tolerance=float(config["intervention"]["norm_relative_tolerance"]),
                projection_tolerance=float(
                    config["intervention"]["realized_span_projection_max"]
                ),
                correction_iterations=int(
                    config["intervention"]["correction_iterations"]
                ),
                correction_damping=float(config["intervention"]["correction_damping"]),
                binary_search_steps=int(config["intervention"]["binary_search_steps"]),
            )
            dictionary, inverse = (
                value.cuda() for value in patcher.geometry[layer]
            )
            actual, error, projection, _ = patcher._geometry_match(
                live_current,
                candidates,
                target_norm,
                dictionary,
                inverse,
            )
            del actual
            eligible = (error <= patcher.norm_tolerance) & (
                projection <= patcher.projection_tolerance
            )
            norm_valid = error <= patcher.norm_tolerance
            repaired, repair_error, repair_projection, repair_steps = (
                patcher._lattice_pair_repair(
                    live_current,
                    live_patcher.deltas[layer],
                    target_norm,
                    dictionary,
                    inverse,
                )
            )
            del repaired
            rows.append({
                "kind": kind,
                "arm": arm,
                "candidate_states": int(candidates.shape[0]),
                "joint_passes": int(eligible.sum()),
                "norm_valid_states": int(norm_valid.sum()),
                "min_projection_among_norm_valid": float(
                    projection[norm_valid].min() if bool(norm_valid.any()) else float("inf")
                ),
                "min_joint_objective": float(torch.maximum(
                    error / patcher.norm_tolerance,
                    projection / patcher.projection_tolerance,
                ).min()),
                "pair_repair_pass": bool(
                    repair_error <= patcher.norm_tolerance
                    and repair_projection <= patcher.projection_tolerance
                ),
                "pair_repair_steps": repair_steps,
                "pair_repair_norm_error": repair_error,
                "pair_repair_projection": repair_projection,
                "pair_repair_objective": max(
                    repair_error / patcher.norm_tolerance,
                    repair_projection / patcher.projection_tolerance,
                ),
            })
    import json

    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

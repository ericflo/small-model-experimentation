from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import torch
import yaml


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from branch_geometry import (  # noqa: E402
    balanced_j_branches,
    geometry_receipt,
    gram_matched_non_j,
)
from model_ops import FixedBranchPatcher  # noqa: E402
from task_data import task_prompt  # noqa: E402


def _config():
    return yaml.safe_load((EXP / "configs" / "default.yaml").read_text())


def _lens():
    return torch.load(EXP / "assets" / "context_lens.pt", map_location="cpu", weights_only=True)


def test_balanced_j_and_non_j_geometry_for_every_layer_and_alpha():
    config = _config()
    lens = _lens()
    for layer in config["lens"]["band"]:
        directions = lens["directions"][layer]
        for alpha in config["lens"]["alpha_multipliers"]:
            j = balanced_j_branches(
                directions,
                public_concepts=config["lens"]["public_alias_concepts"],
                target_rms_norm=config["lens"]["replicated_median_delta_norms"][layer] * alpha,
            )
            non_j = gram_matched_non_j(
                directions,
                j,
                seed=config["seeds"]["non_j_geometry"] + layer,
                rtol=config["lens"]["pseudoinverse_rtol"],
            )
            receipt = geometry_receipt(
                directions, j, non_j, rtol=config["lens"]["pseudoinverse_rtol"]
            )
            assert receipt["j"]["width"] == 12
            assert receipt["j"]["rank"] == 11
            assert receipt["j"]["maximum_sum_abs"] <= config["controls"]["branch_sum_norm_max"]
            assert receipt["non_j"]["maximum_sum_abs"] <= config["controls"]["branch_sum_norm_max"]
            assert receipt["gram_relative_error"] <= config["controls"]["gram_relative_error_max"]
            assert receipt["non_j_max_span_projection_fraction"] < 1e-5


def test_geometry_is_deterministic():
    config = _config()
    directions = _lens()["directions"][4]
    j = balanced_j_branches(directions, public_concepts=12, target_rms_norm=2.0)
    a = gram_matched_non_j(directions, j, seed=17, rtol=1e-5)
    b = gram_matched_non_j(directions, j, seed=17, rtol=1e-5)
    assert torch.equal(a, b)


def test_task_prompt_cannot_see_hidden_or_target_fields():
    config = _config()
    task = json.loads((EXP / "data" / "procedural" / "qualification.jsonl").read_text().splitlines()[0])
    aliases = config["data"]["operation_aliases"]
    expected = task_prompt(task, aliases)
    mutated = copy.deepcopy(task)
    mutated["first_op"] = "forbidden"
    mutated["target_pipeline"] = [{"name": "forbidden", "parameter": 999}]
    mutated["hidden"] = [{"input": [999], "output": [888]}]
    assert task_prompt(mutated, aliases) == expected


def test_data_manifest_is_fresh_and_complete():
    config = _config()
    manifest = json.loads((EXP / "data" / "procedural" / "manifest.json").read_text())
    assert manifest["all_disjoint"] is True
    assert manifest["total_new_unique_fingerprints"] == 76
    assert manifest["splits"]["mechanics"]["rows"] == config["data"]["mechanics_tasks"]
    assert manifest["splits"]["qualification"]["rows"] == config["data"]["qualification_tasks"]
    assert manifest["splits"]["confirmation"]["rows"] == config["data"]["confirmation_tasks"]


def test_fixed_branch_patcher_applies_once_at_exact_position():
    layer = torch.nn.Identity()
    branches = torch.tensor([[1.0, -1.0], [0.5, -0.5], [0.0, 0.0]])
    hidden = torch.zeros(2, 4, 3, dtype=torch.bfloat16)
    with FixedBranchPatcher([layer], position=2, branches_by_layer={0: branches}) as patcher:
        output = layer(hidden)
    assert patcher.applications == {0: 1}
    assert torch.equal(output[:, :2], hidden[:, :2])
    assert torch.equal(output[:, 3], hidden[:, 3])
    assert torch.equal(output[:, 2].float(), branches.T)


def test_fixed_branch_patcher_rejects_wrong_batch_width():
    layer = torch.nn.Identity()
    branches = torch.zeros(3, 2)
    hidden = torch.zeros(1, 4, 3)
    try:
        with FixedBranchPatcher([layer], position=2, branches_by_layer={0: branches}):
            layer(hidden)
    except RuntimeError as error:
        assert "batch width" in str(error)
    else:
        raise AssertionError("wrong branch batch width did not fail")

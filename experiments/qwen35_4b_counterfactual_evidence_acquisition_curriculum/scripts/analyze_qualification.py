#!/usr/bin/env python3
"""Apply frozen two-block acquisition-headroom and reachability gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "src"))

import harness  # noqa: E402
import repo_tasks  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(
    path: Path,
    cfg: dict,
    block: str,
    contract: str,
    scenario_set: str,
    expected_answer_max_tokens: int,
) -> dict:
    payload = json.loads(path.read_text())
    expected = (block, contract, scenario_set, "deep")
    observed = (
        payload.get("block"),
        payload.get("contract"),
        payload.get("scenario_set"),
        payload.get("mode"),
    )
    if observed != expected:
        raise SystemExit(f"wrong qualification receipt {path}: {observed} != {expected}")
    if payload.get("arm") != "start":
        raise SystemExit(f"qualification receipt is not the registered start arm: {path}")
    if payload.get("model_weight_sha256") != cfg["model"]["start_weight_sha256"]:
        raise SystemExit(f"qualification receipt has the wrong model weights: {path}")
    try:
        model_path = Path(payload["model"]).resolve()
        registered_start = Path(cfg["model"]["start_checkpoint"])
        if not registered_start.is_absolute():
            registered_start = ROOT / registered_start
        if model_path != registered_start.resolve():
            raise ValueError("model path is not the frozen start checkpoint")
        expected_control_hashes = {
            "model_config_sha256": cfg["model"]["start_config_sha256"],
            "model_generation_config_sha256": cfg["model"][
                "start_generation_config_sha256"
            ],
            "merge_receipt_sha256": cfg["model"][
                "start_merge_receipt_sha256"
            ],
        }
        control_files = {
            "model_config_sha256": model_path / "config.json",
            "model_generation_config_sha256": (
                model_path / "generation_config.json"
            ),
            "merge_receipt_sha256": model_path / "merge_receipt.json",
        }
        for key, expected_hash in expected_control_hashes.items():
            if (
                not control_files[key].is_file()
                or sha256_file(control_files[key]) != expected_hash
                or payload.get(key) != expected_hash
            ):
                raise ValueError(f"start checkpoint control drift at {key}")
        tokenizer = harness.validate_registered_tokenizer_provenance(
            model_path, payload
        )
        merge_receipt = json.loads(
            (model_path / "merge_receipt.json").read_text(encoding="utf-8")
        )
        registered = harness.validate_registered_tokenizer_provenance(
            model_path, merge_receipt, allow_absent=True
        )
    except (KeyError, TypeError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"qualification tokenizer provenance drifted: {path}: {exc}"
        ) from exc
    if tokenizer != registered:
        raise SystemExit(f"qualification/merge tokenizer mismatch: {path}")
    if (
        tokenizer["tokenizer_manifest_sha256"]
        != cfg["model"]["start_tokenizer_manifest_sha256"]
        or tokenizer["tokenizer_compatibility_sha256"]
        != cfg["model"]["tokenizer_compatibility_sha256"]
    ):
        raise SystemExit(f"qualification tokenizer is not the frozen start identity: {path}")
    if payload.get("scaffold") is not False:
        raise SystemExit(f"qualification receipt unexpectedly uses a scaffold: {path}")
    if payload.get("history_policy") != "canonical_first_valid_tool_call":
        raise SystemExit(f"qualification history policy drifted: {path}")
    if payload.get("answer_max_tokens") != expected_answer_max_tokens:
        raise SystemExit(f"qualification answer allowance drifted: {path}")
    block_cfg = cfg["evaluation"]["blocks"][block]
    registered_n = int(block_cfg["tasks_per_family"])
    if payload.get("tasks_per_family") != registered_n:
        raise SystemExit(f"qualification task count override detected: {path}")
    expected_tasks = repo_tasks.make_pairs(
        tuple(cfg["families"][block_cfg["families"]]),
        registered_n // 2,
        int(block_cfg["seed"]),
        block,
        explicit_contract=contract == "explicit",
    )
    expected_content = [repo_tasks.content_digest(task) for task in expected_tasks]
    expected_manifests = {
        "task_manifest_sha256": repo_tasks.manifest_digest(expected_tasks),
        "task_content_manifest_sha256": hashlib.sha256(
            json.dumps(sorted(expected_content), separators=(",", ":")).encode()
        ).hexdigest(),
        "pair_static_manifest_sha256": hashlib.sha256(
            json.dumps(
                sorted(repo_tasks.pair_static_digest(task) for task in expected_tasks),
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
    }
    for key, expected_value in expected_manifests.items():
        if payload.get(key) != expected_value:
            raise SystemExit(f"qualification manifest drift at {key}: {path}")
    expected_mapping = [
        {
            "task_id": task.task_id,
            "pair_id": task.pair_id,
            "branch": task.branch,
            "family": task.family,
            "evidence_channel": task.evidence_channel,
            "evidence_path": task.evidence_path,
            "evidence_path_regime": task.evidence_path_regime,
            "acquisition_query_skin": task.acquisition_query_skin,
            "evidence_sha256": hashlib.sha256(
                task.files[task.evidence_path].encode()
            ).hexdigest(),
            "acquisition_query_sha256": hashlib.sha256(
                task.acquisition_query.encode()
            ).hexdigest(),
            "oracle_first_patch_sha256": hashlib.sha256(
                json.dumps(
                    task.oracle_patches[0].__dict__, sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        }
        for task in expected_tasks
    ]
    if payload.get("composed_mapping_manifest") != expected_mapping:
        raise SystemExit(f"qualification composed mapping drifted: {path}")
    expected_cases = len(expected_tasks)
    aggregate = payload.get("aggregate", {})
    if (
        aggregate.get("n_cases") != expected_cases
        or aggregate.get("n_dyads") != expected_cases // 2
    ):
        raise SystemExit(f"qualification case/dyad count drifted: {path}")
    code_hashes = {
        "config_sha256": sha256_file(EXP / "configs" / "default.yaml"),
        "evaluator_sha256": sha256_file(EXP / "scripts" / "eval_repo_agent.py"),
        "repo_agent_sha256": sha256_file(EXP / "src" / "repo_agent.py"),
        "task_generator_sha256": sha256_file(EXP / "src" / "repo_tasks.py"),
    }
    for key, expected_value in code_hashes.items():
        if payload.get(key) != expected_value:
            raise SystemExit(f"qualification code/config drift at {key}: {path}")
    summaries = payload.get("runner_summaries") or []
    if not summaries:
        raise SystemExit(f"qualification receipt has no runner summaries: {path}")
    expected_runner_sha = sha256_file(EXP / "src" / "vllm_runner.py")
    if {row.get("runner_sha256") for row in summaries} != {expected_runner_sha}:
        raise SystemExit(f"qualification runner hash drifted: {path}")
    engine = cfg["engine"]
    for row in summaries:
        actual_engine = row.get("engine") or {}
        for key in (
            "max_model_len", "gpu_memory_utilization", "max_num_seqs",
            "max_num_batched_tokens",
        ):
            if actual_engine.get(key) != engine[key]:
                raise SystemExit(f"qualification engine drift at {key}: {path}")
        sampling = row.get("sampling") or {}
        if (
            sampling.get("thinking_budget") != cfg["evaluation"]["think_budget"]
            or sampling.get("answer_max_tokens") != expected_answer_max_tokens
        ):
            raise SystemExit(f"qualification sampling budget drifted: {path}")
    trajectory_sampled = sum(
        row["sampled_tokens"] for row in payload.get("trajectories", [])
    )
    summary_sampled = sum(row["counts"]["sampled_tokens"] for row in summaries)
    trajectory_input = sum(
        row["logical_model_input_tokens"] for row in payload.get("trajectories", [])
    )
    summary_input = sum(
        row["counts"]["logical_model_input_tokens"] for row in summaries
    )
    if trajectory_sampled != summary_sampled or trajectory_input != summary_input:
        raise SystemExit(f"qualification token accounting mismatch: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXP / "configs" / "default.yaml")
    parser.add_argument("--interface-receipt", type=Path, required=True)
    for block in ("a", "b"):
        parser.add_argument(f"--unassisted-{block}", type=Path, required=True)
        parser.add_argument(f"--injected-{block}", type=Path, required=True)
        parser.add_argument(f"--control-search-{block}", type=Path, required=True)
        parser.add_argument(f"--explicit-{block}", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    gates = cfg["qualification_gates"]
    interface = json.loads(args.interface_receipt.read_text())
    if (
        args.interface_receipt.resolve()
        != (EXP / "analysis" / "interface_answer_band.json").resolve()
        or interface.get("schema_version") != 1
        or interface.get("stage") != "interface_answer_band_selection"
        or interface.get("selector_sha256")
        != sha256_file(EXP / "scripts" / "select_interface_band.py")
        or interface.get("config_sha256") != sha256_file(args.config)
        or not interface.get("gate", {}).get("passed")
        or interface.get("qualification_authorized") is not True
        or interface.get("training_authorized") is not False
        or interface.get("menagerie_authorized") is not False
    ):
        raise SystemExit("interface band did not authorize qualification")
    rung_receipts = interface.get("rung_receipts")
    rung_rows = interface.get("rungs")
    if (
        not isinstance(rung_receipts, list)
        or not isinstance(rung_rows, list)
        or len(rung_receipts) != len(rung_rows)
        or not rung_receipts
    ):
        raise SystemExit("interface selection has no rooted rung receipt chain")
    for registration, row in zip(rung_receipts, rung_rows, strict=True):
        if not isinstance(registration, dict) or set(registration) != {"path", "sha256"}:
            raise SystemExit("malformed interface rung registration")
        rung_path = Path(registration["path"])
        if (
            not rung_path.is_file()
            or sha256_file(rung_path) != registration["sha256"]
            or json.loads(rung_path.read_text()) != row
        ):
            raise SystemExit(f"interface rung receipt changed: {rung_path}")
    selected_answer_max_tokens = int(interface["selected_answer_max_tokens"])
    if selected_answer_max_tokens not in cfg["evaluation"]["interface_answer_rungs"]:
        raise SystemExit("interface receipt selected an unregistered answer rung")
    blocks = {}
    all_payloads = []
    for short, name in (("a", "qualification_a"), ("b", "qualification_b")):
        unassisted = load(
            getattr(args, f"unassisted_{short}"), cfg, name, "inferred", "acquisition",
            selected_answer_max_tokens,
        )
        injected = load(
            getattr(args, f"injected_{short}"), cfg, name, "inferred", "injected",
            selected_answer_max_tokens,
        )
        control_search = load(
            getattr(args, f"control_search_{short}"), cfg, name, "inferred", "random",
            selected_answer_max_tokens,
        )
        explicit = load(
            getattr(args, f"explicit_{short}"), cfg, name, "explicit", "acquisition",
            selected_answer_max_tokens,
        )
        for comparator in (injected, control_search):
            for key in (
                "task_manifest_sha256", "task_content_manifest_sha256",
                "pair_static_manifest_sha256", "composed_mapping_manifest",
            ):
                if unassisted.get(key) != comparator.get(key):
                    raise SystemExit(f"inferred-arm task mismatch at {key}: {name}")
        all_payloads.extend((unassisted, injected, control_search, explicit))
        u = unassisted["aggregate"]
        i = injected["aggregate"]
        r = control_search["aggregate"]
        e = explicit["aggregate"]
        supported = [
            family for family, row in i["per_family"].items()
            if row["paired_preverifier_success"]
            >= float(gates["supported_family_injected_success_min"])
        ]
        required_channels = set(gates["required_channels"])
        supported_channels = {
            channel for channel, row in i["per_channel"].items()
            if row["paired_preverifier_success"]
            >= float(gates["supported_channel_injected_success_min"])
        }
        required_query_skins = set(gates["required_qualification_query_skins"])
        supported_query_skins = {
            skin for skin, row in i["per_query_skin"].items()
            if row["paired_preverifier_success"]
            >= float(gates["supported_query_skin_injected_success_min"])
        }
        checks = {
            "unassisted_acquisition_headroom": (
                u["evidence_acquired_before_first_patch"]
                <= float(gates["unassisted_evidence_acquisition_max"])
            ),
            "unassisted_paired_headroom": (
                u["paired_preverifier_success"]
                <= float(gates["unassisted_paired_preverifier_success_max"])
            ),
            "injected_reachability": (
                i["paired_preverifier_success"]
                >= float(gates["evidence_injected_paired_preverifier_success_min"])
            ),
            "injected_delta": (
                i["paired_preverifier_success"] - u["paired_preverifier_success"]
                >= float(gates["evidence_injected_delta_vs_unassisted_min"])
            ),
            "injected_specificity_vs_nondiscriminating_search": (
                i["paired_preverifier_success"] - r["paired_preverifier_success"]
                >= float(gates["evidence_injected_delta_vs_nondiscriminating_search_min"])
            ),
            "explicit_control": (
                e["first_patch_full_correct"]
                >= float(gates["explicit_first_patch_full_correct_min"])
            ),
            "explicit_conditionality": (
                e["unnecessary_evidence_before_first_patch"]
                <= float(gates["explicit_unnecessary_evidence_max"])
            ),
            "supported_family_count": (
                len(supported) >= int(gates["minimum_supported_families"])
            ),
            "all_channels_reachable": (
                set(i["per_channel"]) == required_channels
                and supported_channels == required_channels
            ),
            "all_qualification_query_skins_reachable": (
                set(i["per_query_skin"]) == required_query_skins
                and supported_query_skins == required_query_skins
            ),
        }
        blocks[name] = {
            "task_manifest_sha256": unassisted["task_manifest_sha256"],
            "task_content_manifest_sha256": unassisted["task_content_manifest_sha256"],
            "unassisted": {
                "paired_preverifier_success": u["paired_preverifier_success"],
                "evidence_acquired_before_first_patch": u["evidence_acquired_before_first_patch"],
                "first_patch_full_correct": u["first_patch_full_correct"],
                "terminal_success": u["success"],
            },
            "injected": {
                "paired_preverifier_success": i["paired_preverifier_success"],
                "first_patch_full_correct": i["first_patch_full_correct"],
                "terminal_success": i["success"],
                "delta_vs_unassisted": (
                    i["paired_preverifier_success"] - u["paired_preverifier_success"]
                ),
            },
            "nondiscriminating_search": {
                "paired_preverifier_success": r["paired_preverifier_success"],
                "first_patch_full_correct": r["first_patch_full_correct"],
                "terminal_success": r["success"],
            },
            "explicit": {
                "first_patch_full_correct": e["first_patch_full_correct"],
                "unnecessary_evidence_before_first_patch": e[
                    "unnecessary_evidence_before_first_patch"
                ],
            },
            "supported_families": supported,
            "supported_channels": sorted(supported_channels),
            "supported_query_skins": sorted(supported_query_skins),
            "checks": checks,
        }

    interface_checks = {
        "invalid_actions": all(
            payload["aggregate"]["invalid_action_rate_per_turn"]
            <= float(gates["invalid_action_rate_per_turn_max"])
            for payload in all_payloads
        ),
        "answer_limit_contact": all(
            payload["aggregate"]["answer_cap_hit_rate_per_turn"]
            <= float(gates["answer_cap_hit_rate_per_turn_max"])
            for payload in all_payloads
        ),
        "content_disjoint": (
            blocks["qualification_a"]["task_content_manifest_sha256"]
            != blocks["qualification_b"]["task_content_manifest_sha256"]
        ),
    }
    block_checks = {
        name: all(row["checks"].values()) for name, row in blocks.items()
    }
    passed = all(interface_checks.values()) and all(block_checks.values())
    explicit_failed = any(
        not row["checks"]["explicit_control"]
        or not row["checks"]["explicit_conditionality"]
        for row in blocks.values()
    )
    injected_failed = any(
        not row["checks"]["injected_reachability"]
        or not row["checks"]["injected_delta"]
        or not row["checks"]["injected_specificity_vs_nondiscriminating_search"]
        or not row["checks"]["supported_family_count"]
        or not row["checks"]["all_channels_reachable"]
        or not row["checks"]["all_qualification_query_skins_reachable"]
        for row in blocks.values()
    )
    headroom_failed = any(
        not row["checks"]["unassisted_acquisition_headroom"]
        or not row["checks"]["unassisted_paired_headroom"]
        for row in blocks.values()
    )
    if not all(interface_checks.values()) or explicit_failed:
        verdict = "INSTRUMENT_FAIL"
    elif injected_failed:
        verdict = "NO_INJECTED_REACHABILITY"
    elif headroom_failed:
        verdict = "NO_ACQUISITION_HEADROOM"
    else:
        verdict = "ACQUISITION_QUALIFIED"
    result = {
        "schema_version": 1,
        "stage": "counterfactual_evidence_acquisition_qualification",
        "analyzer_sha256": sha256_file(Path(__file__).resolve()),
        "config_sha256": sha256_file(args.config),
        "model": cfg["model"],
        "interface_receipt": {
            "path": str(args.interface_receipt.resolve()),
            "sha256": sha256_file(args.interface_receipt),
        },
        "input_receipts": {
            f"{short}_{label}": {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for short in ("a", "b")
            for label, path in (
                ("unassisted", getattr(args, f"unassisted_{short}")),
                ("injected", getattr(args, f"injected_{short}")),
                ("control_search", getattr(args, f"control_search_{short}")),
                ("explicit", getattr(args, f"explicit_{short}")),
            )
        },
        "selected_answer_max_tokens": selected_answer_max_tokens,
        "blocks": blocks,
        "interface_checks": interface_checks,
        "block_checks": block_checks,
        "gate": {"passed": passed, "verdict": verdict},
        "training_authorized": passed,
        "menagerie_authorized": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 4


if __name__ == "__main__":
    raise SystemExit(main())

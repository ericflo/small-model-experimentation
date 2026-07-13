#!/usr/bin/env python3
"""Prove transfer margins remain attainable before opening later candidate arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from analyze_transfer import validate_matcher_receipt
from downstream_common import (
    EXP,
    assert_behavior_peers,
    fail,
    read_json,
    sha256_file,
    validate_behavior_receipt,
    validate_selected_answer_allowance,
)


def baseline_headroom_checks(
    cfg: dict,
    *,
    start: float,
    incumbent: float,
    explicit_control: float,
    shuffled_control: float,
    start_normal: float,
) -> dict[str, bool]:
    gates = cfg["transfer_gates"]
    upper = 1.0
    return {
        "delta_vs_start_attainable": upper - start
        >= float(gates["paired_preverifier_delta_vs_start_min"]),
        "delta_vs_incumbent_attainable": upper - incumbent
        >= float(gates["paired_preverifier_delta_vs_incumbent_min"]),
        "delta_vs_explicit_control_attainable": upper - explicit_control
        >= float(gates["paired_preverifier_delta_vs_controls_min"]),
        "delta_vs_shuffled_control_attainable": upper - shuffled_control
        >= float(gates["paired_preverifier_delta_vs_controls_min"]),
        "normal_delta_attainable": upper - start_normal
        >= float(gates["normal_terminal_delta_vs_start_min"]),
        "injected_absolute_attainable": upper
        >= float(gates["injected_paired_preverifier_success_min"]),
        "rejected_transition_attainable": upper
        >= float(gates["rejected_transition_absolute_min"]),
        "failed_transition_attainable": upper
        >= float(gates["failed_transition_absolute_min"]),
        "verified_attainable": upper >= float(gates["verified_absolute_min"]),
        "commit_attainable": upper >= float(gates["commit_absolute_min"]),
        "explicit_retention_attainable": upper
        >= float(gates["explicit_first_patch_retention_min"]),
        "channel_floor_attainable": upper
        >= float(gates["per_channel_paired_preverifier_min"]),
        "query_floor_attainable": upper
        >= float(gates["per_query_skin_paired_preverifier_min"]),
    }


def comparator_headroom_checks(
    cfg: dict,
    *,
    nondiscriminating_search: float,
    sample_start: float | None,
    sample_incumbent: float | None,
) -> dict[str, bool]:
    gates = cfg["transfer_gates"]
    upper = 1.0
    return {
        "delta_vs_nondiscriminating_search_attainable": (
            upper - nondiscriminating_search
            >= float(
                gates["paired_preverifier_delta_vs_nondiscriminating_search_min"]
            )
        ),
        "delta_vs_stronger_sample_more_attainable": (
            sample_start is not None
            and sample_incumbent is not None
            and upper - max(sample_start, sample_incumbent)
            >= float(gates["paired_preverifier_delta_vs_sample_more_min"])
        ),
        "sample_pool_compute_attainable": (
            sample_start is not None and sample_incumbent is not None
        ),
    }


def _behavior(
    path: Path,
    cfg: dict,
    *,
    block: str,
    arm: str,
    weight: str,
    scenario: str = "acquisition",
) -> dict:
    return validate_behavior_receipt(
        path,
        cfg,
        block=block,
        contract="inferred",
        scenario_set=scenario,
        mode="deep",
        arm=arm,
        expected_weight_sha256=weight,
        scaffold=False,
    )


def _rate(payload: dict, key: str = "paired_preverifier_success") -> float:
    return float(payload["aggregate"][key])


def _require_paths(args: argparse.Namespace, names: tuple[str, ...]) -> None:
    missing = [name for name in names if getattr(args, name) is None]
    if missing:
        fail(f"transfer feasibility {args.phase} phase is missing: {missing}")


def _matcher_rate_or_infeasible(
    path: Path,
    cfg: dict,
    *,
    block: str,
    candidate_path: Path,
    candidate: dict,
    pool_arm: str,
    pool_weight_sha256: str,
) -> float | None:
    """Validate either a successful prefix match or an authenticated exhaustion."""
    raw = read_json(path)
    if raw.get("status") == "PASS":
        receipt = validate_matcher_receipt(
            path,
            cfg,
            block=block,
            candidate_path=candidate_path,
            candidate=candidate,
            pool_arm=pool_arm,
            pool_weight_sha256=pool_weight_sha256,
        )
        return float(receipt["dual_overmatch_paired_preverifier_success"])
    try:
        pool_path = Path(raw["sample_pool"])
        target_path = Path(raw["target"])
    except (KeyError, TypeError) as exc:
        fail(f"malformed failed sample matcher {path}: {exc}")
    if (
        raw.get("schema_version") != 1
        or raw.get("status") != "FAIL"
        or raw.get("reason") != "sample_pool_compute_infeasible"
        or raw.get("analyzer_sha256")
        != sha256_file(EXP / "scripts" / "analyze_sample_pool.py")
        or target_path.resolve() != candidate_path.resolve()
        or raw.get("target_receipt_sha256") != sha256_file(candidate_path)
        or raw.get("target_arm") != "candidate"
        or raw.get("target_model_weight_sha256")
        != candidate["model_weight_sha256"]
        or raw.get("pool_arm") != pool_arm
        or raw.get("pool_model_weight_sha256") != pool_weight_sha256
        or not pool_path.is_file()
        or raw.get("sample_pool_receipt_sha256") != sha256_file(pool_path)
    ):
        fail(f"failed sample matcher has stale or mismatched provenance: {path}")
    validate_behavior_receipt(
        pool_path,
        cfg,
        block=block,
        contract="inferred",
        scenario_set="acquisition",
        mode="sample_pool",
        arm=pool_arm,
        expected_weight_sha256=pool_weight_sha256,
        scaffold=False,
    )
    cases = raw.get("cases")
    exhausted = raw.get("exhausted_case_ids")
    if (
        not isinstance(cases, list)
        or not isinstance(exhausted, list)
        or not exhausted
        or exhausted
        != [
            row.get("case_id")
            for row in cases
            if row.get("match", {}).get("pool_exhausted") is True
        ]
    ):
        fail(f"failed sample matcher does not prove pool exhaustion: {path}")
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXP / "configs" / "default.yaml")
    parser.add_argument("--block", choices=["transfer_dev", "transfer_confirm"], required=True)
    parser.add_argument("--phase", choices=["baseline", "comparators"], required=True)
    parser.add_argument("--start", type=Path)
    parser.add_argument("--incumbent", type=Path)
    parser.add_argument("--explicit-control", type=Path)
    parser.add_argument("--shuffled-control", type=Path)
    parser.add_argument("--start-normal", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--control-search", type=Path)
    parser.add_argument("--sample-match-start", type=Path)
    parser.add_argument("--sample-match-incumbent", type=Path)
    parser.add_argument("--baseline-feasibility", type=Path)
    parser.add_argument("--expected-candidate-weight-sha256")
    parser.add_argument("--expected-explicit-control-weight-sha256")
    parser.add_argument("--expected-shuffled-control-weight-sha256")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    canonical_config = (EXP / "configs" / "default.yaml").resolve()
    if args.config.resolve() != canonical_config:
        fail("transfer feasibility requires the frozen default config")
    expected_out = EXP / "analysis" / (
        f"{args.block}_baseline_feasibility.json"
        if args.phase == "baseline"
        else f"{args.block}_feasibility.json"
    )
    if args.out.resolve() != expected_out.resolve():
        fail("transfer feasibility output path is not registered")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cfg["__analysis_config_path__"] = str(args.config.resolve())
    model_cfg = cfg["model"]
    receipts: dict[str, Path] = {}

    if args.phase == "baseline":
        names = (
            "start",
            "incumbent",
            "explicit_control",
            "shuffled_control",
            "start_normal",
            "expected_explicit_control_weight_sha256",
            "expected_shuffled_control_weight_sha256",
        )
        _require_paths(args, names)
        start = _behavior(
            args.start, cfg, block=args.block, arm="start",
            weight=model_cfg["start_weight_sha256"],
        )
        incumbent = _behavior(
            args.incumbent, cfg, block=args.block, arm="incumbent",
            weight=model_cfg["anchor_weight_sha256"],
        )
        explicit = _behavior(
            args.explicit_control, cfg, block=args.block,
            arm="explicit_redundant",
            weight=args.expected_explicit_control_weight_sha256,
        )
        shuffled = _behavior(
            args.shuffled_control, cfg, block=args.block,
            arm="shuffled_binding",
            weight=args.expected_shuffled_control_weight_sha256,
        )
        start_normal = _behavior(
            args.start_normal, cfg, block=args.block, arm="start",
            weight=model_cfg["start_weight_sha256"], scenario="normal",
        )
        assert_behavior_peers(
            {
                "start": start,
                "incumbent": incumbent,
                "explicit": explicit,
                "shuffled": shuffled,
            }
        )
        answer_max_tokens = validate_selected_answer_allowance(
            (start, incumbent, explicit, shuffled, start_normal)
        )
        rates = {
            "start": _rate(start),
            "incumbent": _rate(incumbent),
            "explicit_control": _rate(explicit),
            "shuffled_control": _rate(shuffled),
            "start_normal": _rate(start_normal, "success"),
        }
        checks = baseline_headroom_checks(cfg, **rates)
        receipts = {
            "start": args.start,
            "incumbent": args.incumbent,
            "explicit_control": args.explicit_control,
            "shuffled_control": args.shuffled_control,
            "start_normal": args.start_normal,
        }
        ancestor = None
    else:
        names = (
            "candidate",
            "control_search",
            "sample_match_start",
            "sample_match_incumbent",
            "baseline_feasibility",
            "expected_candidate_weight_sha256",
            "expected_explicit_control_weight_sha256",
            "expected_shuffled_control_weight_sha256",
        )
        _require_paths(args, names)
        expected_baseline = EXP / "analysis" / f"{args.block}_baseline_feasibility.json"
        if args.baseline_feasibility.resolve() != expected_baseline.resolve():
            fail("comparator feasibility has the wrong baseline ancestor")
        ancestor = json.loads(args.baseline_feasibility.read_text(encoding="utf-8"))
        if (
            ancestor.get("schema_version") != 1
            or ancestor.get("stage") != "transfer_feasibility"
            or ancestor.get("phase") != "baseline"
            or ancestor.get("block") != args.block
            or ancestor.get("gate", {}).get("passed") is not True
            or ancestor.get("analyzer_sha256") != sha256_file(Path(__file__).resolve())
            or ancestor.get("config_sha256") != sha256_file(args.config)
        ):
            fail("comparator feasibility baseline ancestor is stale or failed")
        ancestor_receipts = ancestor.get("receipts")
        expected_ancestor_names = {
            "start", "incumbent", "explicit_control", "shuffled_control",
            "start_normal",
        }
        if (
            not isinstance(ancestor_receipts, dict)
            or set(ancestor_receipts) != expected_ancestor_names
        ):
            fail("comparator feasibility baseline receipt map is incomplete")
        ancestor_paths = {}
        for name, registration in ancestor_receipts.items():
            try:
                path = Path(registration["path"])
            except (KeyError, TypeError) as exc:
                fail(f"malformed baseline feasibility registration {name}: {exc}")
            if (
                not path.is_file()
                or registration.get("sha256") != sha256_file(path)
            ):
                fail(f"stale baseline feasibility input: {name}")
            ancestor_paths[name] = path
        ancestor_start = _behavior(
            ancestor_paths["start"], cfg, block=args.block, arm="start",
            weight=model_cfg["start_weight_sha256"],
        )
        ancestor_incumbent = _behavior(
            ancestor_paths["incumbent"], cfg, block=args.block, arm="incumbent",
            weight=model_cfg["anchor_weight_sha256"],
        )
        ancestor_explicit = _behavior(
            ancestor_paths["explicit_control"], cfg, block=args.block,
            arm="explicit_redundant",
            weight=args.expected_explicit_control_weight_sha256,
        )
        ancestor_shuffled = _behavior(
            ancestor_paths["shuffled_control"], cfg, block=args.block,
            arm="shuffled_binding",
            weight=args.expected_shuffled_control_weight_sha256,
        )
        ancestor_normal = _behavior(
            ancestor_paths["start_normal"], cfg, block=args.block, arm="start",
            weight=model_cfg["start_weight_sha256"], scenario="normal",
        )
        assert_behavior_peers({
            "start": ancestor_start,
            "incumbent": ancestor_incumbent,
            "explicit": ancestor_explicit,
            "shuffled": ancestor_shuffled,
        })
        validate_selected_answer_allowance((
            ancestor_start, ancestor_incumbent, ancestor_explicit,
            ancestor_shuffled, ancestor_normal,
        ))
        recomputed_rates = {
            "start": _rate(ancestor_start),
            "incumbent": _rate(ancestor_incumbent),
            "explicit_control": _rate(ancestor_explicit),
            "shuffled_control": _rate(ancestor_shuffled),
            "start_normal": _rate(ancestor_normal, "success"),
        }
        if (
            ancestor.get("rates") != recomputed_rates
            or ancestor.get("checks")
            != baseline_headroom_checks(cfg, **recomputed_rates)
        ):
            fail("comparator feasibility baseline does not recompute")
        candidate = _behavior(
            args.candidate, cfg, block=args.block, arm="candidate",
            weight=args.expected_candidate_weight_sha256,
        )
        control = _behavior(
            args.control_search, cfg, block=args.block, arm="candidate",
            weight=args.expected_candidate_weight_sha256, scenario="random",
        )
        validate_selected_answer_allowance((candidate, control))
        start_rate = _matcher_rate_or_infeasible(
            args.sample_match_start,
            cfg,
            block=args.block,
            candidate_path=args.candidate,
            candidate=candidate,
            pool_arm="start",
            pool_weight_sha256=model_cfg["start_weight_sha256"],
        )
        incumbent_rate = _matcher_rate_or_infeasible(
            args.sample_match_incumbent,
            cfg,
            block=args.block,
            candidate_path=args.candidate,
            candidate=candidate,
            pool_arm="incumbent",
            pool_weight_sha256=model_cfg["anchor_weight_sha256"],
        )
        rates = {
            "nondiscriminating_search": _rate(control),
            "sample_start": start_rate,
            "sample_incumbent": incumbent_rate,
        }
        checks = comparator_headroom_checks(cfg, **rates)
        answer_max_tokens = int(candidate["answer_max_tokens"])
        receipts = {
            "candidate": args.candidate,
            "control_search": args.control_search,
            "sample_match_start": args.sample_match_start,
            "sample_match_incumbent": args.sample_match_incumbent,
            "baseline_feasibility": args.baseline_feasibility,
        }

    passed = all(checks.values())
    prefix = args.block.upper()
    result = {
        "schema_version": 1,
        "stage": "transfer_feasibility",
        "phase": args.phase,
        "block": args.block,
        "answer_max_tokens": answer_max_tokens,
        "rates": rates,
        "checks": checks,
        "gate": {
            "passed": passed,
            "verdict": f"{prefix}_FEASIBLE" if passed else f"{prefix}_INFEASIBLE",
        },
        "candidate_exposure": args.phase == "comparators",
        "receipts": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in receipts.items()
        },
        "analyzer_sha256": sha256_file(Path(__file__).resolve()),
        "config_sha256": sha256_file(args.config),
        "menagerie_authorized": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 4


if __name__ == "__main__":
    raise SystemExit(main())

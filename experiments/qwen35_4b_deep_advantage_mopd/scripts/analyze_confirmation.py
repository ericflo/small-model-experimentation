#!/usr/bin/env python3
"""Apply the complete preregistered two-block procedural decision rule."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


EXP = Path(__file__).resolve().parents[1]
EVALUATOR = EXP / "scripts" / "eval_policy.py"
sys.path.insert(0, str(EXP / "src"))
sys.path.insert(0, str(EXP / "scripts"))

from bench import (  # noqa: E402
    _publication_start_state,
    _publish_json_no_clobber,
)

from confirmation_artifacts import (  # noqa: E402
    confirmation_admission_binding,
    configured_confirmation_raw_root,
    controls_authorization_binding,
    validate_confirmation_campaign_tree,
    validate_confirmation_geometry,
    validate_confirmation_score_artifacts,
)
from confirmation_protocol import (  # noqa: E402
    canonical_backend_protocol,
    expected_confirmation_sampling_protocol,
    validate_confirmation_sampling_protocol,
)
from io_utils import (  # noqa: E402
    confirmation_evaluator_source_inventory,
    load_config,
    sha256_file,
)


def _analysis_publication_start(path: Path) -> tuple[Path, bool]:
    """Freeze the sole canonical analysis destination and its initial state."""

    return _publication_start_state(
        path,
        expected=EXP / "analysis" / "confirmation.json",
        label="confirmation analysis output",
    )


def _publish_analysis_no_clobber(
    path: Path, result: dict, *, existed_at_start: bool
) -> bool:
    """Atomically seal analysis; an exact pre-existing result is read-only."""

    return _publish_json_no_clobber(
        path,
        result,
        expected=EXP / "analysis" / "confirmation.json",
        label="confirmation analysis output",
        existed_at_start=existed_at_start,
    )


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _cell(row: dict) -> str:
    return f"{row['stratum']}/{row['family']}/{row['kind']}/L{int(row['level'])}"


def _item_map(payload: dict) -> dict[str, dict]:
    rows = {str(row["key"]): row for row in payload["items"]}
    if len(rows) != len(payload["items"]):
        raise ValueError(f"duplicate item key in arm {payload.get('tag')}")
    return rows


def _macro_scores(rows: Iterable[tuple[dict, float]]) -> dict[str, float]:
    cells: dict[str, list[float]] = defaultdict(list)
    metadata = {}
    for row, value in rows:
        if not math.isfinite(float(value)):
            raise ValueError("non-finite confirmation score")
        key = _cell(row)
        cells[key].append(float(value))
        metadata[key] = str(row["stratum"])
    by_stratum = {}
    for stratum in ("quick", "deep"):
        values = [
            sum(cells[key]) / len(cells[key])
            for key in sorted(cells)
            if metadata[key] == stratum
        ]
        if not values:
            raise ValueError(f"confirmation has no {stratum} cells")
        by_stratum[stratum] = sum(values) / len(values)
    return {
        "quick": by_stratum["quick"],
        "deep": by_stratum["deep"],
        "joint": 0.5 * (by_stratum["quick"] + by_stratum["deep"]),
    }


def _payload_scores(payload: dict) -> dict[str, float]:
    return _macro_scores((row, float(row["score"])) for row in payload["items"])


def _paired_rows(primary: dict, comparator: dict) -> list[dict]:
    left = _item_map(primary)
    right = _item_map(comparator)
    if set(left) != set(right):
        raise ValueError("confirmation arms are not item-paired")
    rows = []
    for key in sorted(left):
        a, b = left[key], right[key]
        identity = (a["family"], a["kind"], int(a["level"]), a["stratum"])
        other = (b["family"], b["kind"], int(b["level"]), b["stratum"])
        if identity != other:
            raise ValueError(f"paired item metadata differs at {key}")
        rows.append({**a, "delta": float(a["score"]) - float(b["score"])})
    return rows


def _comparison(
    primary_blocks: list[dict],
    comparator_blocks: list[dict],
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> dict:
    if len(primary_blocks) != len(comparator_blocks) or not primary_blocks:
        raise ValueError("comparison block geometry mismatch")
    block_rows = [
        _paired_rows(primary, comparator)
        for primary, comparator in zip(primary_blocks, comparator_blocks)
    ]
    block_scores = [
        _macro_scores((row, row["delta"]) for row in rows) for rows in block_rows
    ]
    pooled = {
        key: sum(score[key] for score in block_scores) / len(block_scores)
        for key in ("quick", "deep", "joint")
    }
    cells: dict[tuple[int, str], list[float]] = defaultdict(list)
    cell_strata = {}
    for block_index, rows in enumerate(block_rows):
        for row in rows:
            key = (block_index, _cell(row))
            cells[key].append(float(row["delta"]))
            cell_strata[key] = str(row["stratum"])
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        per_block = []
        for block_index in range(len(block_rows)):
            stratum_means = {}
            for stratum in ("quick", "deep"):
                means = []
                for key in sorted(cells):
                    if key[0] != block_index or cell_strata[key] != stratum:
                        continue
                    values = cells[key]
                    means.append(
                        sum(values[rng.randrange(len(values))] for _ in values) / len(values)
                    )
                if not means:
                    raise ValueError(f"bootstrap block {block_index} lacks {stratum}")
                stratum_means[stratum] = sum(means) / len(means)
            per_block.append(0.5 * (stratum_means["quick"] + stratum_means["deep"]))
        draws.append(sum(per_block) / len(per_block))
    draws.sort()
    index = max(0, min(len(draws) - 1, math.floor((1.0 - confidence) * len(draws))))
    lcb = float(draws[index])
    return {
        "block_macro_deltas": [
            {**score, "block_seed": int(primary_blocks[index]["block_seed"])}
            for index, score in enumerate(block_scores)
        ],
        "pooled_macro_delta": pooled,
        "one_sided_joint_lcb": lcb,
        "all_block_joint_means_positive": all(score["joint"] > 0.0 for score in block_scores),
        "positive_joint_mean": pooled["joint"] > 0.0,
        "positive_joint_lcb": lcb > 0.0,
        "paired_items_per_block": [len(rows) for rows in block_rows],
    }


def _visible_router(quick: dict, deep: dict, tag: str) -> dict:
    quick_rows, deep_rows = _item_map(quick), _item_map(deep)
    if set(quick_rows) != set(deep_rows):
        raise ValueError("source arms cannot form a paired visible router")
    items = []
    for key in sorted(quick_rows):
        quick_row, deep_row = quick_rows[key], deep_rows[key]
        chosen = quick_row if quick_row["stratum"] == "quick" else deep_row
        items.append(dict(chosen))
    return {
        "stage": "derived_visible_router",
        "tag": tag,
        "scope": "confirmatory",
        "block_seed": int(quick["block_seed"]),
        "decode": "greedy",
        "items": items,
    }


def _cross_score_protocol_checks(
    arms: dict[str, list[dict]], *, block_count: int, engine: dict
) -> dict[str, bool]:
    """Recompute campaign-wide backend and exact-task pairing invariants."""

    recomputed = []
    for payloads in arms.values():
        for payload in payloads:
            backend, fingerprint = canonical_backend_protocol(
                payload.get("runner_summary"),
                expected_engine=engine,
                expected_model=str(payload.get("model", "")),
            )
            recomputed.append((payload, backend, fingerprint))
    checks = {
        "all_backend_fingerprints_recomputed": all(
            payload.get("backend_protocol") == backend
            and payload.get("backend_fingerprint") == fingerprint
            for payload, backend, fingerprint in recomputed
        ),
        "one_exact_backend_across_all_scores": len(
            {fingerprint for _, _, fingerprint in recomputed}
        )
        == 1,
    }
    for block_index in range(block_count):
        task_hashes = {
            payloads[block_index].get("task_manifest_sha256")
            for payloads in arms.values()
        }
        plan_hashes = {
            payloads[block_index].get("ordered_plan_sha256")
            for payloads in arms.values()
        }
        checks[f"block_{block_index}_exact_task_manifest_paired"] = (
            len(task_hashes) == 1 and None not in task_hashes
        )
        checks[f"block_{block_index}_ordered_plan_paired"] = (
            len(plan_hashes) == 1 and None not in plan_hashes
        )
    return checks


def _require_policy_score_protocol(
    payload: dict, *, config: dict, block_seed: int
) -> None:
    """Reject non-policy or decode-labelled artifacts before analysis."""

    engine_protocol = payload.get("engine_protocol")
    if (
        payload.get("stage") != "policy_eval"
        or not isinstance(engine_protocol, dict)
        or not engine_protocol
        or any(value is not True for value in engine_protocol.values())
    ):
        raise ValueError("confirmation analysis requires authenticated policy_eval scores")
    expected_sampling = expected_confirmation_sampling_protocol(
        config,
        decode=str(payload.get("decode", "")),
        block_seed=block_seed,
    )
    validate_confirmation_sampling_protocol(
        payload.get("runner_summary"), expected_sampling
    )
    if payload.get("sampling_protocol") != expected_sampling:
        raise ValueError("confirmation analysis sampling protocol is stale")


def _retention_cells(primary_blocks: list[dict], soup_blocks: list[dict], families: set[str]) -> dict:
    values: dict[str, list[float]] = defaultdict(list)
    for primary, soup in zip(primary_blocks, soup_blocks):
        for row in _paired_rows(primary, soup):
            if str(row["family"]) in families:
                values[_cell(row)].append(float(row["delta"]))
    cells = {
        key: {"n": len(rows), "mean_delta": sum(rows) / len(rows)}
        for key, rows in sorted(values.items())
    }
    if not cells:
        raise ValueError("no registered retention-anchor cells in confirmation")
    return cells


def _transfer_families(primary_blocks: list[dict], soup_blocks: list[dict], families: list[str]) -> dict:
    result = {}
    for family in families:
        cell_values: dict[str, list[float]] = defaultdict(list)
        for primary, soup in zip(primary_blocks, soup_blocks):
            for row in _paired_rows(primary, soup):
                if str(row["family"]) == family:
                    cell_values[_cell(row)].append(float(row["delta"]))
        if not cell_values:
            raise ValueError(f"no confirmation cells for transfer family {family}")
        means = [sum(values) / len(values) for values in cell_values.values()]
        result[family] = {
            "cells": len(means),
            "macro_delta": sum(means) / len(means),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, default=EXP / "analysis" / "confirmation.json"
    )
    args = parser.parse_args()
    out, output_existed_at_start = _analysis_publication_start(args.out)
    config, config_path = load_config(args.config)
    raw_root = configured_confirmation_raw_root(config)
    manifest = _load(args.manifest)
    recorded_authorization = manifest.get("controls_authorization")
    if not isinstance(recorded_authorization, dict):
        raise ValueError("confirmation manifest lacks controls authorization")
    authorization_binding = controls_authorization_binding(
        Path(str(recorded_authorization.get("path", ""))),
        expected_config_sha256=sha256_file(config_path),
    )
    recorded_admission = manifest.get("confirmation_admission")
    if not isinstance(recorded_admission, dict):
        raise ValueError("confirmation manifest lacks global admission")
    admission_binding = confirmation_admission_binding(
        Path(str(recorded_admission.get("path", ""))),
        expected_config_sha256=sha256_file(config_path),
        expected_controls_authorization=authorization_binding,
    )
    validate_confirmation_campaign_tree(
        Path(str(admission_binding["path"])),
        raw_root=raw_root,
        terminal=True,
        require_manifest=True,
    )
    expected_seeds = [int(value) for value in config["seeds"]["confirmatory_blocks"]]
    recorded_arms = manifest.get("arms")
    if not isinstance(recorded_arms, dict):
        raise ValueError("confirmation manifest arm inventory is malformed")
    arms = {}
    for name, paths in recorded_arms.items():
        if not isinstance(name, str) or not isinstance(paths, list):
            raise ValueError("confirmation manifest arm inventory is malformed")
        arms[name] = []
        for index, path in enumerate(paths):
            if index >= len(expected_seeds):
                raise ValueError("confirmation arm has too many block artifacts")
            payload = validate_confirmation_score_artifacts(
                Path(path),
                expected_tag=f"block_{index}_{name}",
                raw_root=raw_root,
            )
            validate_confirmation_geometry(payload, config)
            _require_policy_score_protocol(
                payload, config=config, block_seed=expected_seeds[index]
            )
            arms[name].append(payload)
    primary_name = str(manifest["primary_arm"])
    quick_name = str(manifest["quick_arm"])
    deep_name = str(manifest["deep_arm"])
    soup_name = str(manifest["soup_arm"])
    sample_more_name = str(manifest["sample_more_arm"])
    strict_names = [str(value) for value in manifest["strict_comparator_arms"]]
    replicate_names = [str(value) for value in manifest["replicate_arms"]]
    expected_arm_names = {
        primary_name, quick_name, deep_name, soup_name, sample_more_name,
        *strict_names, *replicate_names,
    }
    model_receipts = manifest.get("model_merge_receipts", {})
    model_inventories = manifest.get("model_inference_inventories", {})
    evaluator_source = manifest.get("evaluator_source_inventory", {})
    current_evaluator_source = confirmation_evaluator_source_inventory()
    protocol = {
        "all_raw_artifacts_current": True,
        "all_raw_semantics_match_scores": True,
        "all_confirmation_geometry_exact": True,
        "manifest_evaluator_current": manifest.get("evaluator_sha256")
        == sha256_file(EVALUATOR),
        "manifest_evaluator_source_current": evaluator_source
        == current_evaluator_source,
        "manifest_config_matches": manifest.get("config_sha256") == sha256_file(config_path),
        "manifest_blocks_match": manifest.get("block_seeds") == expected_seeds,
        "manifest_controls_authorization_current": recorded_authorization
        == authorization_binding,
        "manifest_global_admission_current": recorded_admission
        == admission_binding,
        "exact_arm_inventory": set(arms) == expected_arm_names,
        "all_arms_have_two_blocks": all(len(payloads) == len(expected_seeds) for payloads in arms.values()),
        "all_block_seeds_match": all(
            [int(payload["block_seed"]) for payload in payloads] == expected_seeds
            for payloads in arms.values()
        ),
        "all_confirmatory_scope": all(
            payload.get("scope") == "confirmatory"
            for payloads in arms.values() for payload in payloads
        ),
        "all_scores_are_authenticated_policy_evals": all(
            payload.get("stage") == "policy_eval"
            for payloads in arms.values()
            for payload in payloads
        ),
        "all_sampling_protocols_match_registration": all(
            payload.get("sampling_protocol")
            == expected_confirmation_sampling_protocol(
                config,
                decode=str(payload.get("decode", "")),
                block_seed=int(payload.get("block_seed", -1)),
            )
            for payloads in arms.values()
            for payload in payloads
        ),
        "all_scores_postdate_exact_control_admission": all(
            payload.get("controls_authorization") == authorization_binding
            and payload.get("confirmation_admission") == admission_binding
            for payloads in arms.values()
            for payload in payloads
        ),
        "all_evaluators_match_manifest": all(
            payload.get("evaluator_sha256") == manifest.get("evaluator_sha256")
            for payloads in arms.values()
            for payload in payloads
        ),
        "all_evaluator_sources_match_manifest": (
            isinstance(evaluator_source, dict)
            and set(evaluator_source) == {"sha256", "file_count"}
            and all(
                payload.get("evaluator_source_inventory_sha256")
                == evaluator_source.get("sha256")
                and int(payload.get("evaluator_source_file_count", -1))
                == evaluator_source.get("file_count")
                for payloads in arms.values()
                for payload in payloads
            )
        ),
        "all_model_receipts_match_manifest": all(
            payload.get("model_merge_receipt_sha256") == model_receipts.get(name)
            for name, payloads in arms.items() for payload in payloads
        ),
        "all_model_inference_inventories_match_manifest": (
            set(model_inventories) == set(arms)
            and all(
                payload.get("model_inference_inventory_sha256")
                == model_inventories.get(name)
                for name, payloads in arms.items()
                for payload in payloads
            )
        ),
        "greedy_except_sample_more": all(
            payload.get("decode") == ("sample8" if name == sample_more_name else "greedy")
            and int(payload.get("k", -1)) == (
                int(config["controls"]["sample_more_k"]) if name == sample_more_name else 1
            )
            for name, payloads in arms.items() for payload in payloads
        ),
        "all_engine_protocols_pass": all(
            isinstance(payload.get("engine_protocol"), dict)
            and bool(payload["engine_protocol"])
            and all(value is True for value in payload["engine_protocol"].values())
            for payloads in arms.values() for payload in payloads
        ),
        "all_token_ledgers_authenticated": all(
            isinstance(payload.get("token_ledger"), dict)
            and set(payload["token_ledger"]) == {"sampled_tokens"}
            and isinstance(payload["token_ledger"]["sampled_tokens"], int)
            and not isinstance(payload["token_ledger"]["sampled_tokens"], bool)
            and payload["token_ledger"]["sampled_tokens"] >= 0
            for payloads in arms.values()
            for payload in payloads
        ),
    }
    protocol.update(
        _cross_score_protocol_checks(
            arms, block_count=len(expected_seeds), engine=config["engine"]
        )
    )
    for block_index in range(len(expected_seeds)):
        key_sets = {
            tuple(sorted(_item_map(payloads[block_index]))) for payloads in arms.values()
        }
        protocol[f"block_{block_index}_all_arms_item_paired"] = len(key_sets) == 1
    visible_name = "visible_router"
    visible_blocks = [
        _visible_router(arms[quick_name][index], arms[deep_name][index], visible_name)
        for index in range(len(expected_seeds))
    ]

    bootstrap_samples = int(config["confirmation"]["bootstrap_samples"])
    confidence = float(config["confirmation"]["confidence"])
    strict_payloads = {name: arms[name] for name in strict_names}
    strict_payloads[visible_name] = visible_blocks
    comparisons = {
        name: _comparison(
            arms[primary_name], payloads,
            samples=bootstrap_samples,
            confidence=confidence,
            seed=81100 + index,
        )
        for index, (name, payloads) in enumerate(strict_payloads.items())
    }
    sample_more = _comparison(
        arms[primary_name], arms[sample_more_name],
        samples=bootstrap_samples, confidence=confidence, seed=82100,
    )
    replicate_comparisons = {
        replicate: {
            source: _comparison(
                arms[replicate], arms[source],
                samples=bootstrap_samples,
                confidence=confidence,
                seed=83100 + replicate_index * 10 + source_index,
            )
            for source_index, source in enumerate((quick_name, deep_name, soup_name))
        }
        for replicate_index, replicate in enumerate(replicate_names)
    }

    source_scores = {
        source: [_payload_scores(payload) for payload in arms[source]]
        for source in (quick_name, deep_name)
    }
    primary_scores = [_payload_scores(payload) for payload in arms[primary_name]]
    stratum_dominance = []
    for block_index, seed in enumerate(expected_seeds):
        row = {"block_seed": seed}
        for stratum in ("quick", "deep"):
            better_source = max(
                source_scores[quick_name][block_index][stratum],
                source_scores[deep_name][block_index][stratum],
            )
            primary_value = primary_scores[block_index][stratum]
            row[stratum] = {
                "primary": primary_value,
                "better_source": better_source,
                "delta": primary_value - better_source,
                "passed": primary_value > better_source,
            }
        stratum_dominance.append(row)

    maximum_retention = float(config["decision"]["maximum_retention_regression"])
    retention = _retention_cells(
        arms[primary_name], arms[soup_name],
        set(config["strata"]["retention_anchor_families"]),
    )
    retention_passed = all(
        row["mean_delta"] >= -maximum_retention for row in retention.values()
    )
    maximum_transfer = float(config["decision"]["maximum_transfer_regression"])
    transfer = _transfer_families(
        arms[primary_name], arms[soup_name],
        [str(value) for value in config["strata"]["transfer_families"]],
    )
    transfer_passed = all(
        row["macro_delta"] >= -maximum_transfer for row in transfer.values()
    )

    score_table = {
        name: [_payload_scores(payload) for payload in payloads]
        for name, payloads in arms.items()
    }
    score_table[visible_name] = [_payload_scores(payload) for payload in visible_blocks]
    checks = {
        "protocol": all(protocol.values()),
        "primary_positive_lcb_and_both_blocks_vs_sources_controls_router": all(
            row["positive_joint_mean"]
            and row["positive_joint_lcb"]
            and row["all_block_joint_means_positive"]
            for row in comparisons.values()
        ),
        "primary_both_strata_above_better_source_each_block": all(
            row[stratum]["passed"]
            for row in stratum_dominance for stratum in ("quick", "deep")
        ),
        "replicate_seeds_positive_vs_quick_deep_soup_each_block": all(
            comparison["positive_joint_mean"]
            and comparison["all_block_joint_means_positive"]
            for by_source in replicate_comparisons.values()
            for comparison in by_source.values()
        ),
        "retention_cells_within_ceiling": retention_passed,
        "transfer_families_within_ceiling": transfer_passed,
        "primary_greedy_above_soup_best8_both_blocks": (
            sample_more["positive_joint_mean"]
            and sample_more["all_block_joint_means_positive"]
        ),
    }
    passed = all(checks.values())
    result = {
        "schema_version": 2,
        "stage": "two_block_same_prefix_advantage_confirmation",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "controls_authorization": authorization_binding,
        "confirmation_admission": admission_binding,
        "primary_arm": primary_name,
        "protocol_checks": protocol,
        "strict_comparisons": comparisons,
        "sample_more_comparison": sample_more,
        "replicate_comparisons": replicate_comparisons,
        "stratum_dominance": stratum_dominance,
        "retention": {
            "maximum_regression": maximum_retention,
            "cells": retention,
            "passed": retention_passed,
        },
        "transfer": {
            "maximum_regression": maximum_transfer,
            "families": transfer,
            "passed": transfer_passed,
        },
        "score_table": score_table,
        "token_ledgers": {
            name: [payload.get("token_ledger", {}) for payload in payloads]
            for name, payloads in arms.items()
        },
        "checks": checks,
        "gate": {"passed": passed},
        "downstream_authorization": "benchmark_cli" if passed else "stop_before_benchmark_cli",
    }
    _publish_analysis_no_clobber(
        out,
        result,
        existed_at_start=output_existed_at_start,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 4


if __name__ == "__main__":
    raise SystemExit(main())

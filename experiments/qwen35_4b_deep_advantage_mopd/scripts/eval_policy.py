#!/usr/bin/env python3
"""Paired sealed procedural confirmation for one explicit checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

import harness  # noqa: E402
from confirmation_artifacts import (  # noqa: E402
    commit_confirmation_score,
    configured_confirmation_raw_root,
    prepare_confirmation_output,
    validate_confirmation_geometry,
)
from gym.families import load as load_family  # noqa: E402
from io_utils import (  # noqa: E402
    all_families,
    canonical_hash,
    confirmation_evaluator_source_inventory,
    load_config,
    sha256_file,
)


def _counts(config: dict, scope: str) -> tuple[int, int]:
    if scope != "confirmatory":
        raise ValueError("this experiment exposes only the sealed confirmatory scope")
    cfg = config["confirmation"]
    return int(cfg["atoms_per_family_level"]), int(cfg["episodes_per_family_level"])


def _engine_protocol(
    summaries: list[dict], *, engine_cfg: dict, model: Path, model_config_sha256: str
) -> dict[str, bool]:
    """Fail closed unless every generation call used the frozen local engine."""
    capture_sizes = tuple(int(value) for value in engine_cfg["cudagraph_capture_sizes"])
    model_path = str(model.resolve())
    return {
        "summaries_exist": bool(summaries),
        "same_runner": len({row.get("runner_sha256") for row in summaries}) == 1,
        "exact_local_model": all(row.get("model") == model_path for row in summaries),
        "exact_model_config": all(
            row.get("model_config_sha256") == model_config_sha256 for row in summaries
        ),
        "exact_engine_geometry": all(
            int(row.get("engine", {}).get("max_model_len", -1))
            == int(engine_cfg["max_model_len"])
            and float(row.get("engine", {}).get("gpu_memory_utilization", -1.0))
            == float(engine_cfg["gpu_memory_utilization"])
            and int(row.get("engine", {}).get("max_num_seqs", -1))
            == int(engine_cfg["max_num_seqs"])
            and int(row.get("engine", {}).get("max_num_batched_tokens", -1))
            == int(engine_cfg["max_num_batched_tokens"])
            and tuple(row.get("engine", {}).get("cudagraph_capture_sizes") or ())
            == capture_sizes
            for row in summaries
        ),
        "engine_args_match": all(
            int(row.get("engine_args", {}).get("max_num_seqs", -1))
            == int(engine_cfg["max_num_seqs"])
            and int(row.get("engine_args", {}).get("max_num_batched_tokens", -1))
            == int(engine_cfg["max_num_batched_tokens"])
            and tuple(row.get("engine_args", {}).get("cudagraph_capture_sizes") or ())
            == capture_sizes
            for row in summaries
        ),
        "resolved_full_decode_graphs": all(
            tuple(row.get("resolved_cudagraph", {}).get("cudagraph_capture_sizes") or ())
            == capture_sizes
            and int(
                row.get("resolved_cudagraph", {}).get(
                    "max_cudagraph_capture_size", -1
                )
            )
            == capture_sizes[-1]
            and row.get("resolved_cudagraph", {}).get("has_full_cudagraphs") is True
            and row.get("resolved_cudagraph", {}).get("decode_mode") == "FULL"
            for row in summaries
        ),
    }


def _sampled_token_count(atom_rows: list[dict], episode_rows: list[dict]) -> int:
    """Count sampled tokens from the exact slim schemas emitted by the harness."""
    atom_tokens = sum(
        int(output["n_sampled_tokens"])
        for row in atom_rows
        for output in row["outputs"]
    )
    episode_tokens = sum(
        int(turn["n_sampled_tokens"])
        for row in episode_rows
        for turn in row["turns"]
    )
    return atom_tokens + episode_tokens


def _model_inference_inventory_sha256(model: Path) -> str:
    rows = [
        {
            "path": path.relative_to(model).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in sorted(model.rglob("*"))
        if path.is_file()
    ]
    if not rows:
        raise ValueError("local evaluation model has no inference files")
    return canonical_hash(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--scope", choices=("confirmatory",), required=True)
    parser.add_argument("--block-seed", type=int, required=True)
    parser.add_argument("--decode", choices=("greedy", "sample8"), default="greedy")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    evaluator_sha256 = sha256_file(Path(__file__))
    evaluator_source_before = confirmation_evaluator_source_inventory()
    config, config_path = load_config(args.config)
    raw_root = configured_confirmation_raw_root(config)
    families = all_families(config)
    atom_n, episode_n = _counts(config, args.scope)
    k = 1 if args.decode == "greedy" else int(config["controls"]["sample_more_k"])
    greedy = args.decode == "greedy"
    out_dir = args.out_dir
    score_path = out_dir / "scores.json"
    try:
        prepare_confirmation_output(score_path, raw_root=raw_root)
    except ValueError as exc:
        raise SystemExit(f"unsafe confirmation output state: {exc}") from exc

    strata = config["strata"]
    atom_items: list[dict] = []
    item_strata: dict[str, str] = {}
    for family_index, family_name in enumerate(families):
        family = load_family(family_name)
        for level in [*strata["quick_atom_levels"], *strata["deep_atom_levels"]]:
            if int(level) not in family.LEVELS:
                continue
            seed = args.block_seed + family_index * 100_000 + int(level) * 1_000
            generated = family.gen_atoms(seed, int(level), atom_n)
            atom_items.extend(generated)
            for item in generated:
                item_strata[item["id"]] = "quick" if int(level) in strata["quick_atom_levels"] else "deep"

    episode_specs: list[tuple[str, int, int]] = []
    for family_index, family_name in enumerate(families):
        family = load_family(family_name)
        if not getattr(family, "HAS_EPISODES", False):
            continue
        for level in strata["deep_episode_levels"]:
            if int(level) not in family.LEVELS:
                continue
            for index in range(episode_n):
                seed = args.block_seed + 50_000_000 + family_index * 100_000 + int(level) * 1_000 + index
                episode_specs.append((family_name, int(level), seed))

    sampling = dict(
        think_budget=int(config["generation"]["thinking_budget"]),
        answer_max_tokens=int(config["generation"]["answer_max_tokens"]),
        run_seed=args.block_seed,
        greedy=greedy,
        temperature=None if greedy else float(config["confirmation"]["sample_more_temperature"]),
        top_p=None if greedy else float(config["confirmation"]["sample_more_top_p"]),
        top_k=None if greedy else int(config["confirmation"]["sample_more_top_k"]),
    )
    model_path = Path(args.model)
    merge_receipt = model_path / "merge_receipt.json"
    if model_path.exists() and not merge_receipt.is_file():
        raise SystemExit(f"local evaluation model has no merge receipt: {model_path}")
    model_config_sha256 = sha256_file(model_path / "config.json")
    model_inference_inventory_sha256 = _model_inference_inventory_sha256(model_path)
    runner = harness.make_runner(
        config["engine"], model_override=str(model_path.resolve()) if model_path.exists() else args.model
    )
    started = time.perf_counter()
    atom_rows = harness.run_atoms(runner, atom_items, k=k, **sampling)
    episode_rows = harness.run_episodes(runner, episode_specs, k=k, **sampling) if episode_specs else []
    elapsed = time.perf_counter() - started
    metadata = getattr(runner, "eval_summaries", [])
    runner.close()
    evaluator_source_after = confirmation_evaluator_source_inventory()
    if (
        sha256_file(Path(__file__)) != evaluator_sha256
        or evaluator_source_after != evaluator_source_before
        or sha256_file(model_path / "config.json") != model_config_sha256
        or _model_inference_inventory_sha256(model_path)
        != model_inference_inventory_sha256
    ):
        raise SystemExit("evaluation model changed during the confirmation run")
    engine_protocol = _engine_protocol(
        metadata,
        engine_cfg=config["engine"],
        model=model_path,
        model_config_sha256=model_config_sha256,
    )
    if not all(engine_protocol.values()):
        raise SystemExit(f"evaluation engine protocol failed: {engine_protocol}")

    items: list[dict] = []
    token_ledger = defaultdict(int)
    token_ledger["sampled_tokens"] = _sampled_token_count(atom_rows, episode_rows)
    for row in atom_rows:
        outputs = row["outputs"]
        best = max(outputs, key=lambda value: (float(value["score"]), -int(value["sample_index"])))
        items.append({
            "key": row["id"], "family": row["family"], "kind": "atom",
            "level": int(row["level"]), "stratum": item_strata[row["id"]],
            "score": float(best["score"]), "samples": len(outputs),
        })
    grouped: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for row in episode_rows:
        grouped[(row["family"], int(row["level"]), int(row["ep_seed"]))].append(row)
    for (family, level, seed), rows in sorted(grouped.items()):
        best = max(rows, key=lambda value: (float(value["score"]), -int(value["rollout"])))
        items.append({
            "key": f"{family}/episode/L{level}/s{seed}", "family": family,
            "kind": "episode", "level": level, "stratum": "deep",
            "score": float(best["score"]), "samples": len(rows),
        })
    by_stratum = {}
    for stratum in ("quick", "deep"):
        rows = [item for item in items if item["stratum"] == stratum]
        by_stratum[stratum] = {
            "n": len(rows),
            "mean_score": sum(item["score"] for item in rows) / max(1, len(rows)),
        }
    result = {
        "stage": "policy_eval", "tag": args.tag, "scope": args.scope,
        "evaluator_sha256": evaluator_sha256,
        "evaluator_source_inventory_sha256": evaluator_source_before["sha256"],
        "evaluator_source_file_count": evaluator_source_before["file_count"],
        "model": str(model_path.resolve()) if model_path.exists() else args.model,
        "model_merge_receipt_sha256": (
            sha256_file(merge_receipt) if merge_receipt.is_file() else None
        ),
        "model_config_sha256": model_config_sha256,
        "model_inference_inventory_sha256": model_inference_inventory_sha256,
        "config": str(config_path), "config_sha256": sha256_file(config_path),
        "block_seed": args.block_seed,
        "decode": args.decode, "k": k, "families": families,
        "atoms_per_level": atom_n, "episodes_per_level": episode_n,
        "by_stratum": by_stratum, "items": sorted(items, key=lambda value: value["key"]),
        "token_ledger": dict(token_ledger), "wall_seconds": elapsed,
        "engine_protocol": engine_protocol, "runner_summary": metadata,
    }
    try:
        validate_confirmation_geometry(result, config)
        result = commit_confirmation_score(
            score_path,
            result,
            atom_rows=atom_rows,
            episode_rows=episode_rows,
            raw_root=raw_root,
        )
    except ValueError as exc:
        raise SystemExit(f"failed to commit confirmation artifacts: {exc}") from exc
    print(json.dumps({"tag": args.tag, "by_stratum": by_stratum, "tokens": dict(token_ledger)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Command-line runner for Menagerie benchmark tiers."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from harness import SUITE_VERSION
from harness import backends, engine


SCRIPT_DIR = Path(__file__).resolve().parent
TIER_NAMES = ("quick", "medium", "slow", "deep")

# Constants documented from docs/compute_environment.md: per-sequence decode is
# about 12-13 tok/s regardless of batch on this box; model load time is excluded.
PER_SEQ_DECODE_TOKS_PER_S = 12.5
PREFILL_OVERHEAD_S_PER_BATCH = 1.5


def load_tier(name_or_path: str) -> dict:
    """Load a tier JSON by canonical name or path."""

    if "/" not in name_or_path and not name_or_path.endswith(".json"):
        path = SCRIPT_DIR / "tiers" / f"{name_or_path}.json"
    else:
        path = Path(name_or_path)
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def resolve_tier_requests(tier: str | None, estimate: bool) -> list[str]:
    """Resolve CLI tier selection into names or paths."""

    if estimate and (tier is None or tier == "all"):
        return list(TIER_NAMES)
    if tier is None:
        raise ValueError("--tier is required unless --estimate")
    if tier == "all":
        raise ValueError("--tier all is only valid with --estimate")
    return [tier]


def collect_items(families: dict, tier_cfg: dict, seed: int) -> dict[str, list[dict]]:
    """Generate tier items for each family and mode."""

    by_family: dict[str, list[dict]] = {}
    for name, module in families.items():
        items: list[dict] = []
        atom_cfg = tier_cfg["atoms"]
        for level in atom_cfg["levels"]:
            items.extend(module.generate(seed, level, atom_cfg["n_per_level"], "atom"))
        episode_cfg = tier_cfg.get("episodes")
        if episode_cfg is not None:
            for level in episode_cfg["levels"]:
                items.extend(module.generate(seed, level, episode_cfg["n_per_level"], "episode"))
        by_family[name] = items
    return by_family


def run_tier(
    families: dict,
    tier_cfg: dict,
    backend_obj,
    seed: int,
    include_transcripts: bool = True,
    backend_spec: str = "",
    think: bool = False,
    think_budget: int = 512,
) -> dict:
    """Run a tier over already-discovered families with a constructed backend."""

    all_episodes = []
    items_by_family = collect_items(families, tier_cfg, seed)
    episode_cfg = tier_cfg.get("episodes")
    episode_max_turns = episode_cfg.get("max_turns") if episode_cfg else None
    for name, items in items_by_family.items():
        all_episodes.extend(engine.build_episodes(name, families[name], items, episode_max_turns))

    t0 = perf_counter()
    engine_result = engine.run_lockstep(all_episodes, backend_obj.batch_act)
    wall_total_s = perf_counter() - t0

    per_family_scores: dict[str, list[float]] = defaultdict(list)
    for item in engine_result["per_item"]:
        per_family_scores[item["family"]].append(float(item["score"]))

    per_family = {}
    for name in families:
        scores = per_family_scores.get(name, [])
        mean_score = sum(scores) / len(scores) if scores else 0.0
        per_family[name] = {
            "score": mean_score,
            "n": len(scores),
            "wall_s": round(engine_result["per_family_wall_s"].get(name, 0.0), 3),
        }
    aggregate = sum(data["score"] for data in per_family.values()) / len(per_family) if per_family else 0.0

    per_item = engine_result["per_item"]
    if not include_transcripts:
        per_item = [{key: value for key, value in item.items() if key != "transcript"} for item in per_item]

    return {
        "suite_version": SUITE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "tier": tier_cfg["tier"],
        "tier_config": tier_cfg,
        "backend": backend_spec,
        "think": think,
        "think_budget": think_budget,
        "per_family": per_family,
        "aggregate": aggregate,
        "per_item": per_item,
        "rounds": engine_result["rounds"],
        "wall_total_s": wall_total_s,
        "budget_s": tier_cfg["budget_s"],
        "within_budget": wall_total_s <= tier_cfg["budget_s"],
        "backend_stats": dict(getattr(backend_obj, "stats", {})),
    }


def default_out_path(tier_cfg: dict, backend_spec: str, seed: int) -> Path:
    safe_backend = backend_spec.replace(":", "-")
    return SCRIPT_DIR / "results" / f"{tier_cfg['tier']}_{safe_backend}_seed{seed}.json"


def print_result_table(result: dict) -> None:
    """Print a compact per-family score table."""

    print(f"{'family':<18} {'n':>5} {'score':>8} {'wall_s':>8}")
    for name, data in result["per_family"].items():
        print(f"{name:<18} {data['n']:>5} {data['score']:>8.3f} {data['wall_s']:>8.3f}")
    print(f"{'AGGREGATE':<18} {'':>5} {result['aggregate']:>8.3f} {result['wall_total_s']:>8.3f}")
    status = "yes" if result["within_budget"] else "no"
    print(f"wall_total_s={result['wall_total_s']:.3f} budget_s={result['budget_s']} within_budget={status}")


def discover_or_assume_count(families_dir: Path, assume_families: int) -> tuple[int, str]:
    families = engine.discover_families(families_dir)
    if families:
        return len(families), "discovered"
    return assume_families, "assumed"


def estimate_tier(tier_cfg: dict, family_count: int, max_batch: int, think: bool) -> tuple[float, float]:
    token_add = 512 if think else 0
    atom_tokens = tier_cfg["max_new_tokens"]["atom"] + token_add
    episode_tokens = tier_cfg["max_new_tokens"]["episode"] + token_add
    n_atoms = family_count * len(tier_cfg["atoms"]["levels"]) * tier_cfg["atoms"]["n_per_level"]
    episodes_cfg = tier_cfg.get("episodes")
    if episodes_cfg is None:
        n_epis = 0
        episode_rounds = 0
    else:
        n_epis = family_count * len(episodes_cfg["levels"]) * episodes_cfg["n_per_level"]
        episode_rounds = episodes_cfg["max_turns"]

    atom_batches = math.ceil(n_atoms / max_batch) if n_atoms else 0
    episode_batches = math.ceil(n_epis / max_batch) if n_epis else 0
    atom_decode = atom_batches * (atom_tokens / PER_SEQ_DECODE_TOKS_PER_S)
    atom_prefill = atom_batches * PREFILL_OVERHEAD_S_PER_BATCH
    episode_decode = episode_rounds * episode_batches * (
        episode_tokens / PER_SEQ_DECODE_TOKS_PER_S
    )
    episode_prefill = episode_rounds * episode_batches * PREFILL_OVERHEAD_S_PER_BATCH
    atom_time = atom_decode + atom_prefill
    episode_time = episode_decode + episode_prefill
    expected = (atom_decode + episode_decode) * 0.5 + atom_prefill + episode_prefill
    worst = atom_time + episode_time
    return worst, expected


def print_estimates(tier_names: list[str], families_dir: Path, assume_families: int) -> None:
    family_count, source = discover_or_assume_count(families_dir, assume_families)
    if source == "assumed":
        print(f"families: {family_count} (assumption; discovery found 0)")
    else:
        print(f"families: {family_count} (discovered)")
    print("model load time excluded (~60 s once)")
    print(f"{'tier':<8} {'mode':<10} {'batch':>5} {'worst_s':>10} {'expected_s':>11} {'budget_s':>9} {'flag':>8}")
    for tier_request in tier_names:
        tier_cfg = load_tier(tier_request)
        for label, batch, think in (("no-think", 96, False), ("think512", 48, True)):
            worst, expected = estimate_tier(tier_cfg, family_count, batch, think)
            flag = "WITHIN" if worst <= tier_cfg["budget_s"] else "OVER"
            print(
                f"{tier_cfg['tier']:<8} {label:<10} {batch:>5} {worst:>10.1f} "
                f"{expected:>11.1f} {tier_cfg['budget_s']:>9} {flag:>8}"
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Menagerie benchmark tiers.",
        epilog=(
            "The qwen backend must run under "
            "/home/ericflo/Development/small-model-experimentation/.venv/bin/python "
            "(torch cu130 + transformers live there). CPU backends run under any python3."
        ),
    )
    parser.add_argument("--tier", help="quick|medium|slow|deep, a JSON path, or all with --estimate", default=None)
    parser.add_argument("--backend", help="qwen|oracle|random|noisy:EPS|const:TEXT")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", help="output JSON file")
    parser.add_argument("--think", action="store_true", help="enable qwen thinking mode")
    parser.add_argument("--think-budget", type=int, default=512)
    parser.add_argument("--max-batch", type=int, default=None)
    parser.add_argument("--families-dir", default=str(SCRIPT_DIR / "families"))
    parser.add_argument("--no-transcripts", action="store_true")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--estimate", action="store_true", help="print token-math wall-time estimates")
    parser.add_argument("--assume-families", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    max_batch = args.max_batch
    if max_batch is None:
        max_batch = 48 if args.think else 96

    try:
        tier_requests = resolve_tier_requests(args.tier, args.estimate)
    except ValueError as exc:
        parser.error(str(exc))

    families_dir = Path(args.families_dir)
    if args.estimate:
        print_estimates(tier_requests, families_dir, args.assume_families)
        return 0

    if args.backend is None:
        parser.error("--backend is required unless --estimate")
    tier_cfg = load_tier(tier_requests[0])
    families = engine.discover_families(families_dir)
    if not families:
        raise SystemExit(f"no runnable families found in {families_dir}")

    qwen_opts = {
        "model_id": args.model_id,
        "device": args.device,
        "think": args.think,
        "think_budget": args.think_budget,
        "max_batch": max_batch,
        "max_new_tokens": tier_cfg["max_new_tokens"],
    }
    backend_obj = backends.make_backend(args.backend, seed=args.seed, qwen_opts=qwen_opts)
    result = run_tier(
        families,
        tier_cfg,
        backend_obj,
        args.seed,
        include_transcripts=not args.no_transcripts,
        backend_spec=args.backend,
        think=args.think,
        think_budget=args.think_budget,
    )

    out_path = Path(args.out) if args.out else default_out_path(tier_cfg, args.backend, args.seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print_result_table(result)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
